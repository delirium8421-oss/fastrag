import asyncio
import os
import logging
import argparse
import json
from typing import Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
from fast_graphrag import GraphRAG
from fast_graphrag._llm import (
    OpenAILLMService,
    HuggingFaceEmbeddingService,
    OllamaLLMService,
    OllamaEmbeddingService,
)
from fast_graphrag._services import DefaultInformationExtractionService
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration constants
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_path: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: int,
    corpus_fraction: float = None,
    questions_fraction: float = None
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Truncate corpus if fraction specified
    if corpus_fraction is not None:
        if not 0 < corpus_fraction <= 1:
            logging.error(f"❌ Invalid corpus_fraction: {corpus_fraction}. Must be between 0 and 1.")
            return
        original_length = len(context)
        truncate_length = int(original_length * corpus_fraction)
        context = context[:truncate_length]
        logging.info(f"📉 Corpus truncated: {original_length} → {truncate_length} chars ({corpus_fraction*100:.1f}%)")

    # Prepare output directory
    output_dir = f"./results/fast-graphrag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize LLM service based on mode
    if mode == "ollama":
        # Create Ollama LLM service
        llm_service = OllamaLLMService(
            model=model_name,
            base_url=llm_base_url,
        )
        logging.info(f"✅ Using Ollama LLM service: {model_name} at {llm_base_url}")
        
        # Initialize Ollama embedding service
        embedding_service = OllamaEmbeddingService(
            model=embed_model_path,  # This should be the Ollama embedding model name
            base_url=llm_base_url,
            embedding_dim=1024,
        )
        logging.info(f"✅ Initialized Ollama embedding service: {embed_model_path}")
    else:
        # Use API mode with HuggingFace embeddings
        # Initialize embedding model
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
            embedding_model = AutoModel.from_pretrained(embed_model_path)
            logging.info(f"✅ Loaded embedding model: {embed_model_path}")
        except Exception as e:
            logging.error(f"❌ Failed to load embedding model: {e}")
            return
        
        embedding_service = HuggingFaceEmbeddingService(
            model=embedding_model,
            tokenizer=embedding_tokenizer,
            embedding_dim=1024,
            max_token_size=8192
        )
        
        # Initialize LLM service
        llm_service = OpenAILLMService(
            model=model_name,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        logging.info(f"✅ Using OpenAI-compatible LLM service: {model_name} at {llm_base_url}")

    # Initialize GraphRAG with gleaning enabled
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service,
            information_extraction_service_cls=lambda graph_upsert: DefaultInformationExtractionService(
                graph_upsert=graph_upsert,
                max_gleaning_steps=1  # Enable 1 gleaning iteration
            ),
        ),
    )
    
    # Index the corpus content
    grag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return

    # Apply question fraction if specified
    if questions_fraction is not None:
        if not 0 < questions_fraction <= 1:
            logging.error(f"❌ Invalid questions_fraction: {questions_fraction}. Must be between 0 and 1.")
            return
        original_count = len(corpus_questions)
        limit_count = max(1, int(original_count * questions_fraction))
        corpus_questions = corpus_questions[:limit_count]
        logging.info(f"📉 Questions limited: {original_count} → {limit_count} ({questions_fraction*100:.1f}%)")

    # Sample questions if requested (legacy parameter, overrides fraction)
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Processing {len(corpus_questions)} questions for {corpus_name}")
    
    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            # Execute query
            response = grag.query(q["question"])
            context_chunks = response.to_dict()['context']['chunks']
            contexts = [item[0]["content"] for item in context_chunks]
            predicted_answer = response.response

            # Collect results
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": contexts,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            error_msg = str(e)
            logging.error(f"❌ Error processing question {q.get('id')}: {error_msg}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "error": error_msg
            })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    # Define subset paths
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions.parquet"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.parquet",
            "questions": "./Datasets/Questions/novel_questions.parquet"
        }
    }
    
    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./Examples/graphrag_workspace", 
                        help="Base working directory for GraphRAG")
    
    # Model configuration
    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", 
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/bge-large-en-v1.5", 
                        help="HuggingFace model path (for API mode) or Ollama embedding model name (for ollama mode)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")

    # Corpus and questions filtering
    parser.add_argument("--corpus_fraction", type=float, default=None,
                        help="Fraction of corpus to use for indexing (0.0-1.0). E.g., 0.1 = 10%% of corpus")
    parser.add_argument("--questions_fraction", type=float, default=None,
                        help="Fraction of questions to answer (0.0-1.0). E.g., 0.5 = 50%% of questions")

    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"graphrag_{args.subset}.log")
        ]
    )

    # Keep INFO level for main script, only suppress DEBUG from FastGraphRAG library
    # This keeps useful progress logs while hiding verbose debug output
    logging.getLogger("graphrag").setLevel(logging.INFO)
    
    logging.info(f"🚀 Starting GraphRAG processing for subset: {args.subset}")
    
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = []
        for item in corpus_dataset:
            corpus_data.append({
                "corpus_name": item["corpus_name"],
                "context": item["context"]
            })
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Note: corpus_data not filtered here - filtering happens per-corpus in process_corpus()
    # to allow for character-level truncation
    
    # Load question data
    try:
        questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
        question_data = []
        for item in questions_dataset:
            question_data.append({
                "id": item["id"],
                "source": item["source"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence": item["evidence"]
            })
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return
    
    # Process each corpus concurrently using asyncio + threads
    async def _run_all():
        tasks = []
        for item in corpus_data:
            tasks.append(asyncio.to_thread(
                process_corpus,
                item["corpus_name"],
                item["context"],
                args.base_dir,
                args.mode,
                args.model_name,
                args.embed_model_path,
                args.llm_base_url,
                api_key,
                grouped_questions,
                args.sample,
                args.corpus_fraction,
                args.questions_fraction,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception(f"❌ Task failed: {r}")

    asyncio.run(_run_all())

if __name__ == "__main__":
    main()
"""
Hybrid Graph RAG Example
Demonstrates the usage of HybridGraphRAG for processing medical dataset
with better robustness and performance than standard LightRAG.
"""

import asyncio
import os
import logging
import argparse
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm

# Import HybridRAG classes (needed by helper functions)
from hybrid_graph_rag import HybridGraphRAG, HybridRAGFactory

# Lazy imports (datasets only - loaded in main() to speed up startup)
# from datasets import load_dataset

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=False)
load_dotenv(dotenv_path="../../../LightRAG/.env", override=False)  # Also check LightRAG root

# Conditionally import vLLM (optional dependency)
# Try multiple import paths since vLLM might be installed standalone or via LightRAG
VLLM_AVAILABLE = False
VLLM_IMPORT_ERROR = None

try:
    from lightrag.llm.vllm import vllm_model_complete, vllm_embed, cleanup_vllm_resources
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_IMPORT_ERROR = f"lightrag.llm.vllm: {e}"
    try:
        import vllm
        VLLM_AVAILABLE = True
    except ImportError as e2:
        VLLM_IMPORT_ERROR = f"vllm: {e2}"
        VLLM_AVAILABLE = False

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries only in English.

---Goal---
Generate direct and concise one-sentence answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Provide your answer in only one sentence to answer the question adequately.
The sentence structure must be subject, verb and object, active voice.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source corpus."""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


def validate_ollama_server(ollama_url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Ollama server connectivity and check /api/tags endpoint.
    
    Args:
        ollama_url: Base URL of Ollama server (e.g., http://127.0.0.1:8500)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Ensure URL doesn't have trailing slash
        ollama_url = ollama_url.rstrip('/')
        
        # Test connection to /api/tags
        tags_url = f"{ollama_url}/api/tags"
        logger.info(f"🔗 Testing Ollama server at: {tags_url}")
        
        response = requests.get(tags_url, timeout=5)
        
        if response.status_code != 200:
            return False, f"Server returned status {response.status_code}"
        
        # Parse response
        data = response.json()
        
        if "models" not in data:
            return False, "Response missing 'models' field"
        
        models = data.get("models", [])
        
        if not models:
            return False, "No models available on the Ollama server"
        
        logger.info(f"✅ Ollama server is accessible with {len(models)} model(s)")
        return True, None
    
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama server at {ollama_url}"
    except requests.exceptions.Timeout:
        return False, f"Connection timeout for Ollama server at {ollama_url}"
    except json.JSONDecodeError:
        return False, "Invalid JSON response from Ollama server"
    except Exception as e:
        return False, f"Error testing Ollama server: {str(e)}"


def get_available_models(ollama_url: str) -> Optional[List[Dict]]:
    """
    Fetch available models from Ollama server.
    
    Args:
        ollama_url: Base URL of Ollama server
    
    Returns:
        List of model dicts with 'name' field, or None if failed
    """
    try:
        ollama_url = ollama_url.rstrip('/')
        tags_url = f"{ollama_url}/api/tags"
        response = requests.get(tags_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        
        return None
    
    except Exception as e:
        logger.error(f"❌ Failed to get models: {e}")
        return None


def select_model_from_list(models: List[Dict], model_type: str) -> Optional[str]:
    """
    Prompt user to select a model from available options.
    
    Args:
        models: List of available models
        model_type: Type of model ("LLM" or "Embedding")
    
    Returns:
        Selected model name, or None if user cancelled
    """
    if not models:
        logger.error(f"❌ No models available for {model_type}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Available {model_type} Models:")
    print(f"{'='*60}")
    
    for idx, model in enumerate(models, 1):
        model_name = model.get("name", "Unknown")
        details = model.get("details", {})
        param_size = details.get("parameter_size", "Unknown")
        quantization = details.get("quantization_level", "Unknown")
        
        print(f"{idx}. {model_name}")
        print(f"   Size: {param_size}, Quantization: {quantization}")
    
    print(f"\n{'='*60}")
    
    while True:
        try:
            selection = input(f"Select {model_type} model (enter number): ").strip()
            index = int(selection) - 1
            
            if 0 <= index < len(models):
                selected = models[index].get("name")
                logger.info(f"✅ Selected {model_type}: {selected}")
                return selected
            else:
                print(f"❌ Invalid selection. Please enter a number between 1 and {len(models)}")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            logger.info("⚠️  Selection cancelled by user")
            return None


def setup_ollama_mode() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Interactive setup for Ollama mode.
    Prompts user for server URL and model selection.
    
    Returns:
        Tuple of (ollama_url, llm_model, embed_model) or (None, None, None) on failure
    """
    print(f"\n{'='*60}")
    print("🔧 Ollama Mode Configuration")
    print(f"{'='*60}\n")
    
    # Get Ollama server URL from user
    default_url = "http://172.26.208.1:8500"
    ollama_url = input(f"Enter Ollama server URL (default: {default_url}): ").strip()
    
    if not ollama_url:
        ollama_url = default_url
    
    logger.info(f"📍 Using Ollama URL: {ollama_url}")
    
    # Validate server
    is_valid, error_msg = validate_ollama_server(ollama_url)
    
    if not is_valid:
        logger.error(f"❌ Ollama server validation failed: {error_msg}")
        print(f"\n❌ Error: {error_msg}")
        print("Make sure Ollama server is running:")
        print("  ollama serve --port 8500")
        return None, None, None
    
    # Get available models
    models = get_available_models(ollama_url)
    
    if not models:
        logger.error("❌ Failed to fetch models from Ollama server")
        return None, None, None
    
    logger.info(f"📊 Found {len(models)} model(s) on server")
    
    # Check if we have at least 2 models (LLM + Embedding)
    if len(models) < 2:
        logger.error(f"❌ Not enough models: {len(models)} found, need at least 2 (LLM + Embedding)")
        print(f"\n❌ Error: Only {len(models)} model(s) available.")
        print("You need at least 2 models:")
        print("  1. LLM model (e.g., qwen3:1.7b)")
        print("  2. Embedding model (e.g., qwen3-embedding:0.6b)")
        print("\nPull models with:")
        print("  ollama pull qwen3:1.7b")
        print("  ollama pull qwen3-embedding:0.6b")
        return None, None, None
    
    # Let user select LLM model
    print("\n" + "="*60)
    print("Select LLM Model (for text generation)")
    print("="*60)
    llm_model = select_model_from_list(models, "LLM")
    
    if not llm_model:
        logger.error("❌ LLM model selection cancelled")
        return None, None, None
    
    # Let user select Embedding model
    print("\n" + "="*60)
    print("Select Embedding Model (for embeddings)")
    print("="*60)
    embed_model = select_model_from_list(models, "Embedding")
    
    if not embed_model:
        logger.error("❌ Embedding model selection cancelled")
        return None, None, None
    
    logger.info(f"✅ Ollama configuration complete:")
    logger.info(f"   URL: {ollama_url}")
    logger.info(f"   LLM: {llm_model}")
    logger.info(f"   Embedding: {embed_model}")
    
    return ollama_url, llm_model, embed_model



async def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    llm_model: str,
    embed_model: str,
    llm_base_url: str,
    questions: Dict[str, List[dict]],
    retrieve_topk: int = 5,
    use_resource_constrained: bool = False,
    mode: str = "ollama",
    llm_gguf_path: str = None,
    embed_gguf_path: str = None
):
    """Process a single corpus: index it and answer its questions."""
    logger.info(f"📚 Processing corpus: {corpus_name} (mode: {mode})")
    corpus_start_time = time.time()
    
    # Initialize hybrid RAG
    working_dir = os.path.join(base_dir, corpus_name)
    os.makedirs(working_dir, exist_ok=True)
    
    if mode == "vllm":
        if not VLLM_AVAILABLE:
            error_msg = (
                f"vLLM is not available in the current Python environment.\n"
                f"Import error: {VLLM_IMPORT_ERROR}\n"
                f"Make sure you're using the correct Python environment where vLLM is installed.\n"
                f"Install with: pip install vllm or uv pip install vllm"
            )
            logger.error(f"🔧 Debug: VLLM_AVAILABLE={VLLM_AVAILABLE}, Error={VLLM_IMPORT_ERROR}")
            raise ImportError(error_msg)
        
        # For vLLM mode, use the model paths directly (they should be GGUF file paths)
        # Prioritize explicit gguf_path args, otherwise use the model arguments
        final_llm_model = llm_gguf_path if llm_gguf_path else llm_model
        final_embed_model = embed_gguf_path if embed_gguf_path else embed_model
        
        # Also check environment variables as fallback
        final_llm_model = os.getenv("LLM_GGUF_PATH", final_llm_model)
        final_embed_model = os.getenv("EMBED_GGUF_PATH", final_embed_model)
        
        logger.info(f"vLLM mode: LLM from {final_llm_model}, Embedding from {final_embed_model}")
        
        # Use vLLM instance with the provided GGUF models
        hybrid_rag = HybridRAGFactory.create_vllm_instance(
            working_dir=working_dir,
            llm_model=final_llm_model,
            embed_model=final_embed_model
        )
    elif use_resource_constrained:
        hybrid_rag = HybridRAGFactory.create_resource_constrained_instance(
            working_dir=working_dir
        )
    else:
        hybrid_rag = HybridRAGFactory.create_ollama_instance(
            working_dir=working_dir,
            llm_model=llm_model,
            embed_model=embed_model,
            llm_url=llm_base_url,
            llm_model_max_async = 2
        )
    
    # Index the corpus
    index_result = await hybrid_rag.index(context, corpus_name)
    
    if index_result["status"] != "success":
        logger.error(f"❌ Indexing failed: {index_result.get('error')}")
        return None
    
    logger.info(f"✅ Indexing completed: {json.dumps(index_result, indent=2)}")
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logger.warning(f"⚠️  No questions found for corpus: {corpus_name}")
        await hybrid_rag.cleanup()
        return None
    
    logger.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Process questions
    results = []
    query_start_time = time.time()
    
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            answer, context_text = await hybrid_rag.query(
                q["question"],
                top_k=retrieve_topk,
                query_type="hybrid"
            )
            
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": context_text,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": answer,
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            logger.error(f"❌ Query error for question {q.get('id')}: {e}")
            results.append({
                "id": q["id"],
                "error": str(e)
            })
    
    query_time = time.time() - query_start_time
    corpus_time = time.time() - corpus_start_time
    
    # Save results
    output_dir = f"./results/hybrid-rag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved {len(results)} predictions to: {output_path}")
    logger.info(f"⏱️  Query time: {query_time:.2f}s | Total corpus time: {corpus_time:.2f}s")
    
    # Get and log statistics
    stats = hybrid_rag.get_stats()
    logger.info(f"📊 Extraction Statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    await hybrid_rag.cleanup()
    
    return {
        "corpus": corpus_name,
        "results_count": len(results),
        "indexing_time": index_result["indexing_time"],
        "query_time": query_time,
        "total_time": corpus_time,
        "stats": stats
    }


async def main():
    """Main entry point."""
    # Lazy imports - load heavy dependencies after argparse setup for faster startup
    from datasets import load_dataset
    
    # Get script directory for resolving relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define subset paths (absolute paths relative to script location)
    SUBSET_PATHS = {
        "medical": {
            "corpus": os.path.join(script_dir, "..", "Datasets", "Corpus", "medical.parquet"),
            "questions": os.path.join(script_dir, "..", "Datasets", "Questions", "medical_questions.parquet")
        },
        "novel": {
            "corpus": os.path.join(script_dir, "..", "Datasets", "Corpus", "novel.parquet"),
            "questions": os.path.join(script_dir, "..", "Datasets", "Questions", "novel_questions.parquet")
        }
    }
    
    parser = argparse.ArgumentParser(
        description="Hybrid Graph RAG: Process Corpora and Answer Questions"
    )
    
    # Core arguments
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel"],
        help="Subset to process (medical or novel)"
    )
    parser.add_argument(
        "--base_dir",
        default="./hybrid_rag_workspace",
        help="Base working directory"
    )
    
    # Model configuration
    parser.add_argument(
        "--mode",
        default="ollama",
        choices=["ollama", "vllm"],
        help="Inference mode: 'ollama' for Ollama server or 'vllm' for GGUF files"
    )
    parser.add_argument(
        "--llm_model",
        default="qwen3:1.7b",
        help="Ollama LLM model name or path to GGUF file for vLLM mode"
    )
    parser.add_argument(
        "--embed_model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model name or path to GGUF file for vLLM mode"
    )
    parser.add_argument(
        "--llm_base_url",
        default="http://127.0.0.1:8500",
        help="Ollama server URL (for ollama mode)"
    )
    parser.add_argument(
        "--llm_gguf_path",
        default=None,
        help="Path to LLM GGUF file (for vllm mode, overrides LLM_GGUF_PATH env var)"
    )
    parser.add_argument(
        "--embed_gguf_path",
        default=None,
        help="Path to embedding GGUF file (for vllm mode, overrides EMBED_GGUF_PATH env var)"
    )
    parser.add_argument(
        "--retrieve_topk",
        type=int,
        default=5,
        help="Number of top documents to retrieve"
    )
    
    # Sampling arguments
    parser.add_argument(
        "--chunk_sample_percent",
        type=float,
        default=1.0,
        help="Percentage of chunks to process (0-100, default: 0.5%%)"
    )
    parser.add_argument(
        "--question_sample_percent",
        type=float,
        default=100,
        help="Percentage of questions to answer (0-100, default: 100%%)"
    )
    
    # Resource management
    parser.add_argument(
        "--resource_constrained",
        action="store_true",
        help="Use resource-constrained settings (for 4GB GPU)"
    )
    
    args = parser.parse_args()
    
    # If Ollama mode is selected, prompt for server and models
    if args.mode == "ollama":
        logger.info("🔧 Ollama mode detected - initializing interactive setup")
        ollama_url, llm_model, embed_model = setup_ollama_mode()
        
        if ollama_url is None or llm_model is None or embed_model is None:
            logger.error("❌ Ollama setup failed. Exiting.")
            return
        
        # Override command-line defaults with user selections
        args.llm_base_url = ollama_url
        args.llm_model = llm_model
        args.embed_model = embed_model
    
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logger.error(f"❌ Invalid subset: {args.subset}")
        return
    
    # Get file paths
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Create workspace
    os.makedirs(args.base_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Load corpus data
    try:
        logger.info(f"📖 Loading corpus from {corpus_path}")
        corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = []
        for item in corpus_dataset:
            corpus_data.append({
                "corpus_name": item["corpus_name"],
                "context": item["context"]
            })
        logger.info(f"✅ Loaded {len(corpus_data)} corpus documents")
        
        # Sample corpus
        original_len = len(corpus_data)
        sample_ratio = max(0.001, min(1.0, args.chunk_sample_percent / 100.0))
        corpus_data = corpus_data[:max(1, int(len(corpus_data) * sample_ratio))]
        logger.info(f"📊 Sampled to {len(corpus_data)} documents ({args.chunk_sample_percent:.2f}% of {original_len})")
    
    except Exception as e:
        logger.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Load question data
    try:
        logger.info(f"📖 Loading questions from {questions_path}")
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
        logger.info(f"✅ Loaded {len(question_data)} questions")
        
        # Sample questions
        original_q_len = len(question_data)
        q_sample_ratio = max(0.001, min(1.0, args.question_sample_percent / 100.0))
        grouped_questions = {
            k: v[:max(1, int(len(v) * q_sample_ratio))]
            for k, v in grouped_questions.items()
        }
        sampled_q_len = sum(len(v) for v in grouped_questions.values())
        logger.info(f"📊 Sampled to {sampled_q_len} questions ({args.question_sample_percent:.2f}% of {original_q_len})")
    
    except Exception as e:
        logger.error(f"❌ Failed to load questions: {e}")
        return
    
    # Process all corpora
    logger.info(f"🚀 Starting hybrid RAG pipeline (mode={args.mode}, resource_constrained={args.resource_constrained})")
    
    tasks = []
    for item in corpus_data:
        tasks.append(
            process_corpus(
                corpus_name=item["corpus_name"],
                context=item["context"],
                base_dir=args.base_dir,
                llm_model=args.llm_model,
                embed_model=args.embed_model,
                llm_base_url=args.llm_base_url,
                questions=grouped_questions,
                retrieve_topk=args.retrieve_topk,
                use_resource_constrained=args.resource_constrained,
                mode=args.mode,
                llm_gguf_path=args.llm_gguf_path,
                embed_gguf_path=args.embed_gguf_path
            )
        )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Log results
    successful_results = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"❌ Task failed: {r}")
        elif r is not None:
            successful_results.append(r)
            logger.info(f"✅ Completed: {r['corpus']} - {r['results_count']} answers")
    
    total_time = time.time() - start_time
    logger.info(f"🏁 Pipeline completed in {total_time:.2f}s")
    logger.info(f"📊 Processed {len(successful_results)} corpora successfully")


if __name__ == "__main__":
    asyncio.run(main())

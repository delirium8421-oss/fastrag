"""Benchmarking script for GraphRAG."""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import xxhash
from dotenv import load_dotenv
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._llm import gpt_4o_mini_complete
from nano_graphrag._utils import always_get_an_event_loop, logging
from tqdm import tqdm

logging.getLogger("nano-graphrag").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)

@dataclass
class Query:
    """Dataclass for a query."""

    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()


def load_dataset(dataset_name: str, subset: int = 0) -> Any:
    """Load a dataset from the datasets folder."""
    with open(f"./datasets/{dataset_name}.json", "r") as f:
        dataset = json.load(f)

    if subset:
        return dataset[:subset]
    else:
        return dataset


def get_corpus(dataset: Any, dataset_name: str) -> Dict[int, Tuple[int | str, str]]:
    """Get the corpus from the dataset."""
    if dataset_name == "2wikimultihopqa" or dataset_name == "hotpotqa":
        passages: Dict[int, Tuple[int | str, str]] = {}

        for datapoint in dataset:
            context = datapoint["context"]

            for passage in context:
                title, text = passage
                title = title.encode("utf-8").decode()
                text = "\n".join(text).encode("utf-8").decode()
                hash_t = xxhash.xxh3_64_intdigest(text)
                if hash_t not in passages:
                    passages[hash_t] = (title, text)

        return passages
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def get_queries(dataset: Any):
    """Get the queries from the dataset."""
    queries: List[Query] = []

    for datapoint in dataset:
        queries.append(
            Query(
                question=datapoint["question"].encode("utf-8").decode(),
                answer=datapoint["answer"],
                evidence=list(datapoint["supporting_facts"]),
            )
        )

    return queries


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="LightRAG CLI")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use.")
    parser.add_argument("-c", "--create", action="store_true", help="Create the graph for the given dataset.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Benchmark the graph for the given dataset.")
    parser.add_argument("-s", "--score", action="store_true", help="Report scores after benchmarking.")
    parser.add_argument("--mode", default="local", help="LightRAG query mode.")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./db/nano/{args.dataset}_{args.n}"
    corpus = get_corpus(dataset, args.dataset)

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))
        grag = GraphRAG(
            working_dir=working_dir,
            best_model_func=gpt_4o_mini_complete
        )
        grag.insert([f"{title}: {corpus}" for _, (title, corpus) in tuple(corpus.items())])
    if args.benchmark:
        queries = get_queries(dataset)
        print("Dataset loaded. Queries:", len(queries))
        grag = GraphRAG(
            working_dir=working_dir,
            best_model_func=gpt_4o_mini_complete
        )

        async def _query_task(query: Query, mode: str) -> Dict[str, Any]:
            answer = await grag.aquery(
                query.question, QueryParam(mode=mode, only_need_context=True, local_max_token_for_text_unit=9000)
            )
            chunks = []
            for c in re.findall(r"\n-----Sources-----\n```csv\n(.*?)\n```", answer, re.DOTALL)[0].split("\n")[
                1:-1
            ]:
                try:
                    chunks.append(c.split(",\t")[1].split(":")[0].lstrip('"'))
                except IndexError:
                    pass
            return {
                "question": query.question,
                "answer": "",
                "evidence": chunks[:8],
                "ground_truth": [e[0] for e in query.evidence],
            }

        async def _run(mode: str):
            answers = [
                await a
                for a in tqdm(
                    asyncio.as_completed([_query_task(query, mode=mode) for query in queries]), total=len(queries)
                )
            ]
            return answers

        answers = always_get_an_event_loop().run_until_complete(_run(mode=args.mode))

        with open(f"./results/nano/{args.dataset}_{args.n}_{args.mode}.json", "w") as f:
            json.dump(answers, f, indent=4)

    if args.benchmark or args.score:
        with open(f"./results/nano/{args.dataset}_{args.n}_{args.mode}.json", "r") as f:
            answers = json.load(f)

        try:
            with open(f"./questions/{args.dataset}_{args.n}.json", "r") as f:
                questions_multihop = json.load(f)
        except FileNotFoundError:
            questions_multihop = []

        # Compute retrieval metrics
        retrieval_scores: List[float] = []
        retrieval_scores_multihop: List[float] = []

        for answer in answers:
            ground_truth = answer["ground_truth"]
            predicted_evidence = answer["evidence"]

            p_retrieved: float = len(set(ground_truth).intersection(set(predicted_evidence))) / len(set(ground_truth))
            retrieval_scores.append(p_retrieved)

            if answer["question"] in questions_multihop:
                retrieval_scores_multihop.append(p_retrieved)

        print(
            f"Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores])}"
        )
        if len(retrieval_scores_multihop):
            print(
                f"[multihop] Percentage of queries with perfect retrieval: {
                    np.mean([1 if s == 1.0 else 0 for s in retrieval_scores_multihop])
                }"
            )

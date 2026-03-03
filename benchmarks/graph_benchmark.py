"""Benchmarking script for GraphRAG."""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import xxhash
from _domain import DOMAIN, ENTITY_TYPES, QUERIES
from dotenv import load_dotenv
from tqdm import tqdm

from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._utils import get_event_loop


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

    parser = argparse.ArgumentParser(description="GraphRAG CLI")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use.")
    parser.add_argument("-c", "--create", action="store_true", help="Create the graph for the given dataset.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Benchmark the graph for the given dataset.")
    parser.add_argument("-s", "--score", action="store_true", help="Report scores after benchmarking.")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./db/graph/{args.dataset}_{args.n}"
    corpus = get_corpus(dataset, args.dataset)

    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))
        grag = GraphRAG(
            working_dir=working_dir,
            domain=DOMAIN[args.dataset],
            example_queries="\n".join(QUERIES),
            entity_types=ENTITY_TYPES[args.dataset],
        )
        grag.insert(
            [f"{title}: {corpus}" for _, (title, corpus) in tuple(corpus.items())],
            metadata=[{"id": title} for title in tuple(corpus.keys())],
        )
    if args.benchmark:
        queries = get_queries(dataset)
        print("Dataset loaded. Queries:", len(queries))
        grag = GraphRAG(
            working_dir=working_dir,
            domain=DOMAIN[args.dataset],
            example_queries="\n".join(QUERIES),
            entity_types=ENTITY_TYPES[args.dataset],
        )

        async def _query_task(query: Query) -> Dict[str, Any]:
            answer = await grag.async_query(query.question, QueryParam(only_context=True))
            return {
                "question": query.question,
                "answer": answer.response,
                "evidence": [
                    corpus[chunk.metadata["id"]][0]
                        if isinstance(chunk.metadata["id"], int)
                        else chunk.metadata["id"]
                    for chunk, _ in answer.context.chunks
                ],
                "ground_truth": [e[0] for e in query.evidence],
            }

        async def _run():
            await grag.state_manager.query_start()
            answers = [
                await a
                for a in tqdm(asyncio.as_completed([_query_task(query) for query in queries]), total=len(queries))
            ]
            await grag.state_manager.query_done()
            return answers

        answers = get_event_loop().run_until_complete(_run())

        with open(f"./results/graph/{args.dataset}_{args.n}.json", "w") as f:
            json.dump(answers, f, indent=4)

    if args.benchmark or args.score:
        with open(f"./results/graph/{args.dataset}_{args.n}.json", "r") as f:
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

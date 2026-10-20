from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import get_settings
from app.domain import RetrievedChunk
from app.services.container import build_container
from app.services.evaluation import RetrievalMetrics, metrics_at_k
from app.services.retrieval import HybridRetriever


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    question: str
    relevant_ids: frozenset[str]
    relevance_level: Literal["chunk", "document"]


def load_dataset(path: Path) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} must be a JSON object.")
            question = payload.get("question")
            has_chunk_ids = "relevant_chunk_ids" in payload
            has_document_ids = "relevant_document_ids" in payload
            chunk_ids = payload.get("relevant_chunk_ids")
            document_ids = payload.get("relevant_document_ids")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number} requires a non-empty question.")
            if has_chunk_ids == has_document_ids:
                raise ValueError(
                    f"Line {line_number} must include exactly one of relevant_chunk_ids or relevant_document_ids."
                )
            relevant_ids = chunk_ids if has_chunk_ids else document_ids
            if (
                not isinstance(relevant_ids, list)
                or not relevant_ids
                or not all(isinstance(identifier, str) and identifier for identifier in relevant_ids)
            ):
                raise ValueError(f"Line {line_number} relevant IDs must be a non-empty array of strings.")
            examples.append(
                EvaluationExample(
                    question=question.strip(),
                    relevant_ids=frozenset(relevant_ids),
                    relevance_level="chunk" if chunk_ids else "document",
                )
            )
    if not examples:
        raise ValueError("The evaluation dataset contains no examples.")
    return examples


def _ranked_ids(
    results: Sequence[RetrievedChunk], relevance_level: Literal["chunk", "document"]
) -> list[str]:
    if relevance_level == "chunk":
        return [result.chunk.id for result in results]
    # Multiple retrieved chunks from the same file count as one document result.
    return list(dict.fromkeys(result.chunk.document_id for result in results))


def evaluate(examples: list[EvaluationExample], retriever: HybridRetriever, k: int) -> dict[str, object]:
    per_query: list[dict[str, object]] = []
    metrics: list[RetrievalMetrics] = []
    for example in examples:
        retrieved = retriever.retrieve(example.question, top_k=k)
        scores = metrics_at_k(_ranked_ids(retrieved, example.relevance_level), example.relevant_ids, k)
        metrics.append(scores)
        per_query.append(
            {
                "relevance_level": example.relevance_level,
                "retrieved_count": len(retrieved),
                "recall_at_k": scores.recall_at_k,
                "precision_at_k": scores.precision_at_k,
                "reciprocal_rank": scores.reciprocal_rank,
                "ndcg_at_k": scores.ndcg_at_k,
            }
        )
    return {
        "sample_count": len(examples),
        "k": k,
        "mean_recall_at_k": sum(item.recall_at_k for item in metrics) / len(metrics),
        "mean_precision_at_k": sum(item.precision_at_k for item in metrics) / len(metrics),
        "mean_reciprocal_rank": sum(item.reciprocal_rank for item in metrics) / len(metrics),
        "mean_ndcg_at_k": sum(item.ndcg_at_k for item in metrics) / len(metrics),
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a JSONL relevance set.")
    parser.add_argument(
        "dataset", type=Path, help="JSONL file with questions and relevant chunk/document IDs"
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Number of ranked results to evaluate (default: DEFAULT_TOP_K)"
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
        k = args.k if args.k is not None else settings.default_top_k
        if not 1 <= k <= settings.max_top_k:
            raise ValueError(f"--k must be between 1 and MAX_TOP_K ({settings.max_top_k}).")
        examples = load_dataset(args.dataset)
        services = build_container(settings)
        try:
            report = evaluate(examples, services.retriever, k)
        finally:
            services.vectors.close()
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

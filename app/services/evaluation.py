from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def metrics_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
    k: int,
) -> RetrievalMetrics:
    """Compute standard binary-relevance retrieval metrics for the top-k results."""
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_ids:
        raise ValueError("Each evaluation example must have at least one relevant ID")

    top_ids = list(ranked_ids[:k])
    relevance = [identifier in relevant_ids for identifier in top_ids]
    hits = sum(relevance)
    first_hit = next((rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant), None)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(relevance, start=1) if is_relevant)
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return RetrievalMetrics(
        recall_at_k=hits / len(relevant_ids),
        precision_at_k=hits / k,
        reciprocal_rank=(1.0 / first_hit) if first_hit is not None else 0.0,
        ndcg_at_k=(dcg / ideal_dcg) if ideal_dcg else 0.0,
    )

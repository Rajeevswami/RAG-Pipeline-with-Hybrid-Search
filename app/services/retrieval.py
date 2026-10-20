from __future__ import annotations

from collections import defaultdict

from app.config import Settings
from app.domain import RetrievedChunk
from app.services.embeddings import EmbeddingClient
from app.services.reranking import Reranker
from app.storage.qdrant_store import QdrantStore
from app.storage.sqlite_store import SQLiteStore


class HybridRetriever:
    """Dense + BM25 retrieval fused with reciprocal-rank fusion, then reranked."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingClient,
        vectors: QdrantStore,
        lexical: SQLiteStore,
        reranker: Reranker,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vectors = vectors
        self.lexical = lexical
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        candidate_limit = max(self.settings.retrieval_candidates, top_k or 0)
        query_vector = self.embeddings.embed([query])[0]
        dense_hits = self.vectors.search(query_vector, limit=candidate_limit)
        lexical_hits = self.lexical.search_lexical(query, limit=candidate_limit)

        fusion_scores: defaultdict[str, float] = defaultdict(float)
        dense_ranks: dict[str, int] = {}
        lexical_ranks: dict[str, int] = {}
        if self.settings.dense_weight > 0:
            for rank, (chunk_id, _similarity) in enumerate(dense_hits, start=1):
                dense_ranks[chunk_id] = rank
                fusion_scores[chunk_id] += self.settings.dense_weight / (self.settings.rrf_k + rank)
        if self.settings.lexical_weight > 0:
            for rank, chunk_id in enumerate(lexical_hits, start=1):
                lexical_ranks[chunk_id] = rank
                fusion_scores[chunk_id] += self.settings.lexical_weight / (self.settings.rrf_k + rank)

        if not fusion_scores:
            return []
        ordered_ids = sorted(fusion_scores, key=fusion_scores.__getitem__, reverse=True)
        chunk_map = self.lexical.get_chunks_by_ids(ordered_ids)
        candidates = [
            RetrievedChunk(
                chunk=chunk_map[chunk_id],
                fusion_score=fusion_scores[chunk_id],
                dense_rank=dense_ranks.get(chunk_id),
                lexical_rank=lexical_ranks.get(chunk_id),
            )
            for chunk_id in ordered_ids
            if chunk_id in chunk_map
        ]
        reranked = self.reranker.rerank(query, candidates)
        return reranked[: (top_k or self.settings.default_top_k)]

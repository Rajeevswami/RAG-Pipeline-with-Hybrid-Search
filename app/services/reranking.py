from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from app.domain import RetrievedChunk

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "should",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


class Reranker:
    """CrossEncoder reranking when configured; otherwise a light, deterministic reranker."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (model_name or "").strip() or None
        self._cross_encoder: Any | None = None

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if self.model_name:
            return self._cross_encoder_rerank(query, candidates)
        return self._overlap_rerank(query, candidates)

    def _cross_encoder_rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install requirements-reranker.txt to use RERANKER_MODEL.") from exc
            self._cross_encoder = CrossEncoder(self.model_name)
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        reranked = [
            replace(item, rerank_score=float(score)) for item, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)

    @staticmethod
    def _overlap_rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_terms = {
            token.casefold()
            for token in _WORD.findall(query)
            if token.casefold() not in _STOP_WORDS and len(token) > 1
        }
        max_fusion = max((item.fusion_score for item in candidates), default=1.0) or 1.0
        normalized_query = " ".join(_WORD.findall(query)).casefold()
        reranked: list[RetrievedChunk] = []
        for item in candidates:
            words = {token.casefold() for token in _WORD.findall(item.chunk.text)}
            coverage = len(query_terms & words) / max(len(query_terms), 1)
            normalized_text = " ".join(_WORD.findall(item.chunk.text)).casefold()
            phrase_match = 1.0 if normalized_query and normalized_query in normalized_text else 0.0
            scaled_fusion = item.fusion_score / max_fusion
            score = (0.58 * coverage) + (0.32 * scaled_fusion) + (0.10 * phrase_match)
            reranked.append(replace(item, rerank_score=score))
        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)

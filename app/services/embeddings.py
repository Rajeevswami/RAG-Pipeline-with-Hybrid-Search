from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from app.config import Settings


class EmbeddingError(RuntimeError):
    """The configured embedding provider could not create vectors."""


class EmbeddingClient:
    """Dense embedding adapter for OpenAI or an optional local sentence-transformer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._local_model: Any | None = None

    @property
    def dimension(self) -> int:
        if self.settings.embedding_provider == "openai":
            return self.settings.openai_embedding_dimensions
        model = self._get_local_model()
        dimension = model.get_sentence_embedding_dimension()
        if not dimension:
            raise EmbeddingError("The local embedding model did not report its vector dimension.")
        return int(dimension)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.embedding_provider == "openai":
            vectors = self._embed_openai(texts)
        else:
            vectors = self._embed_local(texts)

        expected_dimension = self.dimension
        if len(vectors) != len(texts):
            raise EmbeddingError("The embedding provider returned an incomplete result set.")
        for vector in vectors:
            if len(vector) != expected_dimension:
                raise EmbeddingError(
                    f"The embedding provider returned {len(vector)} dimensions; expected {expected_dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("The embedding provider returned a non-finite vector value.")
        return vectors

    def _embed_openai(self, texts: Sequence[str]) -> list[list[float]]:
        api_key = (self.settings.openai_api_key or "").strip()
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")

        vectors: list[list[float]] = []
        # Keep payloads small and stay below provider request limits for large uploads.
        for start in range(0, len(texts), 64):
            batch = list(texts[start : start + 64])
            body: dict[str, Any] = {
                "model": self.settings.openai_embedding_model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.settings.openai_embedding_model.startswith("text-embedding-3"):
                body["dimensions"] = self.settings.openai_embedding_dimensions
            try:
                response = httpx.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                    timeout=self.settings.request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                items = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
                batch_vectors = [item["embedding"] for item in items]
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                suffix = f" (HTTP {status})" if status else ""
                raise EmbeddingError(f"The OpenAI embeddings request failed{suffix}.") from exc

            if len(batch_vectors) != len(batch):
                raise EmbeddingError("The embedding provider returned an incomplete batch.")
            if any(len(vector) != self.settings.openai_embedding_dimensions for vector in batch_vectors):
                raise EmbeddingError(
                    "The embedding vector size does not match OPENAI_EMBEDDING_DIMENSIONS. "
                    "Use a fresh Qdrant collection when changing embedding dimensions."
                )
            vectors.extend([[float(value) for value in vector] for vector in batch_vectors])
        return vectors

    def _embed_local(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._get_local_model()
        try:
            encoded = model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False,
            )
            return [[float(value) for value in vector] for vector in encoded]
        except Exception as exc:
            raise EmbeddingError("The local sentence-transformer failed to encode the text.") from exc

    def _get_local_model(self) -> Any:
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "Install requirements-local.txt to use EMBEDDING_PROVIDER=sentence-transformers."
                ) from exc
            try:
                self._local_model = SentenceTransformer(self.settings.local_embedding_model)
            except Exception as exc:
                raise EmbeddingError("The configured local embedding model could not be loaded.") from exc
        return self._local_model

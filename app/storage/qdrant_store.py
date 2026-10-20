from __future__ import annotations

import time
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.domain import ChunkRecord


class QdrantStore:
    """Qdrant adapter for dense vectors, with an embedded local-dev mode."""

    def __init__(self, settings: Settings, vector_size: int) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        qdrant_url = (settings.qdrant_url or "").strip()
        if qdrant_url:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=settings.request_timeout_seconds,
            )
            self._wait_for_server()
        else:
            path = Path(settings.qdrant_path)
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

        self._ensure_collection(vector_size)

    def _wait_for_server(self) -> None:
        last_error: Exception | None = None
        for attempt in range(12):
            try:
                self.client.get_collections()
                return
            except Exception as exc:  # Qdrant may still be starting in Compose.
                last_error = exc
                if attempt < 11:
                    time.sleep(1)
        raise RuntimeError(
            "Could not connect to Qdrant. Check QDRANT_URL and the Qdrant service."
        ) from last_error

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            return

        collection = self.client.get_collection(self.collection_name)
        vector_config = collection.config.params.vectors
        if isinstance(vector_config, dict):
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' uses named vectors, but this service expects "
                "one unnamed vector. Configure a new collection before indexing."
            )

        actual_size = getattr(vector_config, "size", None)
        if actual_size is None or int(actual_size) != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' uses {actual_size}-dimensional vectors, "
                f"but the configured embedding provider produces {vector_size}. Configure a new collection "
                "or re-index documents with the matching model."
            )
        if getattr(vector_config, "distance", None) != models.Distance.COSINE:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' must use cosine distance. "
                "Configure a new collection before indexing."
            )

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("The number of Qdrant points must match the number of vectors.")
        for start in range(0, len(chunks), 128):
            batch_chunks = chunks[start : start + 128]
            batch_vectors = vectors[start : start + 128]
            points = [
                models.PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [(str(point.id), float(point.score)) for point in response.points]

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def health_check(self) -> bool:
        self.client.get_collection(self.collection_name)
        return True

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.answering import AnswerService, LLMClient
from app.services.chunking import Chunker
from app.services.embeddings import EmbeddingClient
from app.services.ingestion import IngestionService
from app.services.reranking import Reranker
from app.services.retrieval import HybridRetriever
from app.storage.qdrant_store import QdrantStore
from app.storage.sqlite_store import SQLiteStore


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    lexical: SQLiteStore
    vectors: QdrantStore
    embeddings: EmbeddingClient
    retriever: HybridRetriever
    answers: AnswerService
    ingestion: IngestionService


def build_container(settings: Settings) -> ServiceContainer:
    lexical = SQLiteStore(settings.sqlite_path)
    lexical.initialize()
    embeddings = EmbeddingClient(settings)
    vectors = QdrantStore(settings, embeddings.dimension)
    if settings.embedding_provider == "sentence-transformers":
        chunker = Chunker(
            settings.local_embedding_chunk_size_tokens,
            settings.local_embedding_chunk_overlap_tokens,
        )
    else:
        chunker = Chunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    retriever = HybridRetriever(
        settings=settings,
        embeddings=embeddings,
        vectors=vectors,
        lexical=lexical,
        reranker=Reranker(settings.reranker_model),
    )
    return ServiceContainer(
        settings=settings,
        lexical=lexical,
        vectors=vectors,
        embeddings=embeddings,
        retriever=retriever,
        answers=AnswerService(settings, LLMClient(settings)),
        ingestion=IngestionService(settings, lexical, vectors, embeddings, chunker),
    )

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables or a local .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Atlas | Internal Knowledge"
    environment: str = "development"

    # Persistent metadata and vector stores. Set QDRANT_URL to use a Qdrant server;
    # otherwise the Python client runs a persistent, single-process local instance.
    sqlite_path: Path = Path("data/rag.sqlite3")
    qdrant_url: str | None = None
    qdrant_path: Path = Path("data/qdrant")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "company_docs"

    embedding_provider: Literal["openai", "sentence-transformers"] = "openai"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = Field(default=1536, ge=1)
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_chunk_size_tokens: int = Field(default=160, ge=50, le=1000)
    local_embedding_chunk_overlap_tokens: int = Field(default=32, ge=0, le=200)

    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-4-20250514"
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    # If set, enables a local sentence-transformers CrossEncoder. Without it, a
    # deterministic lexical + reciprocal-rank reranker is used (no model download).
    reranker_model: str | None = None

    max_upload_mb: int = Field(default=25, ge=1, le=100)
    chunk_size_tokens: int = Field(default=420, ge=50, le=2000)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=500)
    retrieval_candidates: int = Field(default=30, ge=1, le=200)
    default_top_k: int = Field(default=6, ge=1, le=20)
    max_top_k: int = Field(default=10, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.55, ge=0)
    lexical_weight: float = Field(default=0.45, ge=0)
    min_citation_quote_chars: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if (
            self.embedding_provider == "sentence-transformers"
            and self.local_embedding_chunk_overlap_tokens >= self.local_embedding_chunk_size_tokens
        ):
            raise ValueError(
                "LOCAL_EMBEDDING_CHUNK_OVERLAP_TOKENS must be smaller than LOCAL_EMBEDDING_CHUNK_SIZE_TOKENS"
            )
        if self.dense_weight + self.lexical_weight <= 0:
            raise ValueError("At least one of DENSE_WEIGHT or LEXICAL_WEIGHT must be positive")
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K cannot exceed MAX_TOP_K")
        return self

    @property
    def has_llm_credentials(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key.strip())
        return bool(self.anthropic_api_key and self.anthropic_api_key.strip())

    @property
    def has_embedding_credentials(self) -> bool:
        if self.embedding_provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key.strip())
        # Local dependencies and model availability are checked on first use.
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    page_number: int | None
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    filename: str
    media_type: str
    checksum: str
    uploaded_at: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    fusion_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float = 0.0

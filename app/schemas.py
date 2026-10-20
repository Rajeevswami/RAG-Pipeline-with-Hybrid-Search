from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    uploaded_at: str
    page_count: int
    chunk_count: int


class EvidenceSource(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    snippet: str
    fusion_score: float
    rerank_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    quote: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=3000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value.strip()


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    citation_verified: bool
    verification_method: str
    citations: list[Citation]
    sources: list[EvidenceSource]
    latency_ms: int


class AppStatus(BaseModel):
    status: str
    app_name: str
    llm_provider: str
    llm_model: str
    llm_configured: bool
    embedding_provider: str
    embedding_configured: bool
    vector_store: str
    reranker: str
    indexed_documents: int

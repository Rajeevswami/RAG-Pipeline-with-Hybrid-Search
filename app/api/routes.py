from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas import AppStatus, DocumentSummary, QueryRequest, QueryResponse
from app.services.answering import LLMConfigurationError, LLMProviderError
from app.services.container import ServiceContainer
from app.services.document_parser import SUPPORTED_EXTENSIONS, DocumentParseError
from app.services.embeddings import EmbeddingError
from app.services.ingestion import DuplicateDocumentError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


ServiceDependency = Annotated[ServiceContainer, Depends(get_services)]
UploadDependency = Annotated[UploadFile, File(...)]


@router.get("/status", response_model=AppStatus)
async def status(services: ServiceDependency) -> AppStatus:
    try:
        await run_in_threadpool(services.lexical.health_check)
        await run_in_threadpool(services.vectors.health_check)
        documents = await run_in_threadpool(services.lexical.list_documents)
        state = "ok"
    except Exception:
        logger.exception("Dependency health check failed")
        documents = []
        state = "degraded"

    if services.settings.llm_provider == "openai":
        model_name = services.settings.openai_chat_model
    else:
        model_name = services.settings.anthropic_model
    vector_store = "Qdrant server" if (services.settings.qdrant_url or "").strip() else "Embedded Qdrant"
    reranker = (
        f"CrossEncoder · {services.settings.reranker_model}"
        if services.settings.reranker_model
        else "Hybrid overlap reranker"
    )
    return AppStatus(
        status=state,
        app_name=services.settings.app_name,
        llm_provider=services.settings.llm_provider,
        llm_model=model_name,
        llm_configured=services.settings.has_llm_credentials,
        embedding_provider=services.settings.embedding_provider,
        embedding_configured=services.settings.has_embedding_credentials,
        vector_store=vector_store,
        reranker=reranker,
        indexed_documents=len(documents),
    )


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(services: ServiceDependency) -> list[DocumentSummary]:
    documents = await run_in_threadpool(services.lexical.list_documents)
    return [DocumentSummary.model_validate(document) for document in documents]


@router.post("/documents", response_model=DocumentSummary, status_code=201)
async def upload_document(
    file: UploadDependency,
    services: ServiceDependency,
) -> DocumentSummary:
    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1].strip()[:255]
    if not filename:
        filename = "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Supported extensions: {allowed}")

    max_bytes = services.settings.max_upload_mb * 1024 * 1024
    try:
        content = await file.read(max_bytes + 1)
    finally:
        await file.close()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {services.settings.max_upload_mb} MB upload limit.",
        )

    try:
        document = await run_in_threadpool(
            services.ingestion.ingest,
            filename,
            file.content_type or "application/octet-stream",
            content,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=503, detail="Document indexing failed. Check the storage and embedding configuration."
        ) from exc
    return DocumentSummary.model_validate(document)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    services: ServiceDependency,
) -> None:
    try:
        deleted = await run_in_threadpool(services.ingestion.delete, document_id)
    except Exception as exc:
        logger.exception("Document deletion failed")
        raise HTTPException(
            status_code=503, detail="Document could not be removed from the indexes."
        ) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    services: ServiceDependency,
) -> QueryResponse:
    started = time.perf_counter()
    top_k = request.top_k or services.settings.default_top_k
    if top_k > services.settings.max_top_k:
        raise HTTPException(
            status_code=422,
            detail=f"top_k cannot exceed the configured maximum of {services.settings.max_top_k}.",
        )

    try:
        retrieved = await run_in_threadpool(services.retriever.retrieve, request.question, top_k)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Hybrid retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Retrieval is temporarily unavailable. Check the embedding and vector-store configuration.",
        ) from exc

    try:
        result = await services.answers.answer(request.question, retrieved)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        logger.warning("Configured answer provider failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Grounded answer generation failed")
        raise HTTPException(
            status_code=502, detail="The answer provider could not produce a verified response."
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    return QueryResponse(
        answer=result.answer,
        abstained=result.abstained,
        citation_verified=result.citation_verified,
        verification_method=result.verification_method,
        citations=result.citations,
        sources=result.sources,
        latency_ms=latency_ms,
    )

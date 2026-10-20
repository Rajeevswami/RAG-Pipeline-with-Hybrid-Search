from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from app.config import Settings
from app.domain import ChunkRecord, DocumentRecord
from app.services.chunking import Chunker
from app.services.document_parser import DocumentParseError, parse_document
from app.services.embeddings import EmbeddingClient
from app.storage.qdrant_store import QdrantStore
from app.storage.sqlite_store import SQLiteStore


class DuplicateDocumentError(ValueError):
    def __init__(self, existing: DocumentRecord) -> None:
        super().__init__(f"This file is already indexed as '{existing.filename}'.")
        self.existing = existing


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        lexical: SQLiteStore,
        vectors: QdrantStore,
        embeddings: EmbeddingClient,
        chunker: Chunker,
    ) -> None:
        self.settings = settings
        self.lexical = lexical
        self.vectors = vectors
        self.embeddings = embeddings
        self.chunker = chunker

    def ingest(self, filename: str, media_type: str, content: bytes) -> DocumentRecord:
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise DocumentParseError(
                f"File is larger than the {self.settings.max_upload_mb} MB upload limit."
            )
        checksum = hashlib.sha256(content).hexdigest()
        existing = self.lexical.find_document_by_checksum(checksum)
        if existing:
            raise DuplicateDocumentError(existing)

        pages = parse_document(filename, content)
        drafts = self.chunker.split_pages(pages)
        if not drafts:
            raise DocumentParseError("No indexable text was found in this document.")

        document_id = str(uuid.uuid4())
        chunks = [
            ChunkRecord(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"atlas-rag:{document_id}:{draft.chunk_index}")),
                document_id=document_id,
                filename=filename,
                chunk_index=draft.chunk_index,
                page_number=draft.page_number,
                text=draft.text,
                token_count=draft.token_count,
            )
            for draft in drafts
        ]
        document = DocumentRecord(
            id=document_id,
            filename=filename,
            media_type=media_type,
            checksum=checksum,
            uploaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            page_count=len(pages),
            chunk_count=len(chunks),
        )

        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        try:
            self.vectors.upsert_chunks(chunks, vectors)
            self.lexical.create_document(document, chunks)
        except Exception:
            # Clean up even if a batched vector upsert failed partway through.
            self.vectors.delete_document(document_id)
            raise
        return document

    def delete(self, document_id: str) -> bool:
        if not self.lexical.get_document(document_id):
            return False
        self.vectors.delete_document(document_id)
        return self.lexical.delete_document(document_id)

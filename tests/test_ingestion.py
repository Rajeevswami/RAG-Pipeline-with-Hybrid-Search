import pytest

from app.config import Settings
from app.services.chunking import Chunker
from app.services.ingestion import DuplicateDocumentError, IngestionService
from app.storage.sqlite_store import SQLiteStore


class FakeEmbeddings:
    def embed(self, texts):
        return [[float(index + 1), 0.0, 0.0] for index, _text in enumerate(texts)]


class FakeVectors:
    def __init__(self):
        self.indexed = {}

    def upsert_chunks(self, chunks, vectors):
        self.indexed.update(
            {chunk.id: (chunk.document_id, vector) for chunk, vector in zip(chunks, vectors, strict=True)}
        )

    def delete_document(self, document_id):
        self.indexed = {
            chunk_id: value for chunk_id, value in self.indexed.items() if value[0] != document_id
        }


@pytest.mark.parametrize("filename", ["policy.md", "policy.txt"])
def test_ingestion_indexes_chunks_in_both_stores(tmp_path, filename) -> None:
    settings = Settings(_env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5)
    lexical = SQLiteStore(tmp_path / "rag.sqlite3")
    lexical.initialize()
    vectors = FakeVectors()
    service = IngestionService(settings, lexical, vectors, FakeEmbeddings(), Chunker(50, 5))
    text = (
        b"Employees may request reimbursement for approved travel expenses. Submit receipts within 30 days."
    )

    document = service.ingest(filename, "text/plain", text)

    assert document.filename == filename
    assert document.chunk_count == 1
    assert lexical.search_lexical("reimbursement", limit=5)
    assert len(vectors.indexed) == 1
    with pytest.raises(DuplicateDocumentError):
        service.ingest(filename, "text/plain", text)

    assert service.delete(document.id) is True
    assert vectors.indexed == {}
    assert lexical.list_documents() == []

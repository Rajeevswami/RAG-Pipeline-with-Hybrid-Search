import uuid

import pytest

from app.config import Settings
from app.domain import ChunkRecord
from app.storage.qdrant_store import QdrantStore


def test_embedded_qdrant_upsert_search_and_delete(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        qdrant_url=None,
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="test_company_docs",
    )
    store = QdrantStore(settings, vector_size=3)
    document_id = str(uuid.uuid4())
    chunk = ChunkRecord(
        id=str(uuid.uuid4()),
        document_id=document_id,
        filename="policy.txt",
        chunk_index=0,
        page_number=None,
        text="The travel policy covers train bookings.",
        token_count=7,
    )
    try:
        store.upsert_chunks([chunk], [[0.1, 0.2, 0.3]])
        hits = store.search([0.1, 0.2, 0.3], limit=5)
        assert hits[0][0] == chunk.id

        store.delete_document(document_id)
        assert store.search([0.1, 0.2, 0.3], limit=5) == []
    finally:
        store.close()


def test_existing_collection_dimension_mismatch_fails_at_startup(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        qdrant_url=None,
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="dimension_test",
    )
    store = QdrantStore(settings, vector_size=3)
    store.close()

    with pytest.raises(RuntimeError, match="uses 3-dimensional vectors"):
        QdrantStore(settings, vector_size=4)

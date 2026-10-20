from app.domain import ChunkRecord, DocumentRecord
from app.storage.sqlite_store import SQLiteStore


def _sample_records() -> tuple[DocumentRecord, list[ChunkRecord]]:
    document = DocumentRecord(
        id="doc-1",
        filename="handbook.txt",
        media_type="text/plain",
        checksum="sha256-example",
        uploaded_at="2026-10-20T12:00:00+00:00",
        page_count=1,
        chunk_count=2,
    )
    chunks = [
        ChunkRecord(
            "chunk-1", "doc-1", "handbook.txt", 0, None, "Employees receive vacation days each year.", 7
        ),
        ChunkRecord(
            "chunk-2", "doc-1", "handbook.txt", 1, None, "Submit receipts through the expense portal.", 7
        ),
    ]
    return document, chunks


def test_document_and_bm25_index_round_trip(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "rag.sqlite3")
    store.initialize()
    document, chunks = _sample_records()

    store.create_document(document, chunks)

    assert store.find_document_by_checksum("sha256-example") == document
    assert store.get_document("doc-1") == document
    assert store.list_documents() == [document]
    assert store.search_lexical("vacation days", limit=5) == ["chunk-1"]
    restored = store.get_chunks_by_ids(["chunk-2"])
    assert restored["chunk-2"].text == chunks[1].text
    assert restored["chunk-2"].filename == "handbook.txt"

    assert store.delete_document("doc-1") is True
    assert store.delete_document("doc-1") is False
    assert store.list_documents() == []
    assert store.search_lexical("vacation", limit=5) == []

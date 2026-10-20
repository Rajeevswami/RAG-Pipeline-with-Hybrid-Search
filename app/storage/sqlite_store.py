from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.domain import ChunkRecord, DocumentRecord

_TOKEN = re.compile(r"[\w]+(?:[-'][\w]+)?", re.UNICODE)


class SQLiteStore:
    """Document metadata and an FTS5-backed BM25 index."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,
                    uploaded_at TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    content,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )

    def create_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        """Atomically save a document, its chunks, and matching FTS rows."""
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO documents
                   (id, filename, media_type, checksum, uploaded_at, page_count, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.id,
                    document.filename,
                    document.media_type,
                    document.checksum,
                    document.uploaded_at,
                    document.page_count,
                    document.chunk_count,
                ),
            )
            connection.executemany(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, page_number, text, token_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.text,
                        chunk.token_count,
                    )
                    for chunk in chunks
                ],
            )
            connection.executemany(
                "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
                [(chunk.id, chunk.document_id, chunk.text) for chunk in chunks],
            )

    def find_document_by_checksum(self, checksum: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE checksum = ?", (checksum,)).fetchone()
        return self._document_from_row(row) if row else None

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC, filename COLLATE NOCASE"
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._connection() as connection:
            exists = connection.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone()
            if not exists:
                return False
            connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return True

    def search_lexical(self, query: str, limit: int) -> list[str]:
        """Return chunk IDs ordered by SQLite FTS5's BM25 score (best first)."""
        match_expression = self._match_expression(query)
        if not match_expression:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT chunk_id
                   FROM chunks_fts
                   WHERE chunks_fts MATCH ?
                   ORDER BY bm25(chunks_fts) ASC
                   LIMIT ?""",
                (match_expression, limit),
            ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""SELECT c.*, d.filename
                    FROM chunks AS c
                    JOIN documents AS d ON d.id = c.document_id
                    WHERE c.id IN ({placeholders})""",
                chunk_ids,
            ).fetchall()
        return {str(row["id"]): self._chunk_from_row(row) for row in rows}

    def health_check(self) -> bool:
        with self._connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return True

    @staticmethod
    def _match_expression(query: str) -> str:
        # Quoting each token prevents user input from becoming FTS5 syntax.
        # OR keeps lexical recall useful for short natural-language questions.
        tokens = list(dict.fromkeys(_TOKEN.findall(query)))[:32]
        return " OR ".join(f'"{token}"' for token in tokens)

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=str(row["id"]),
            filename=str(row["filename"]),
            media_type=str(row["media_type"]),
            checksum=str(row["checksum"]),
            uploaded_at=str(row["uploaded_at"]),
            page_count=int(row["page_count"]),
            chunk_count=int(row["chunk_count"]),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            filename=str(row["filename"]),
            chunk_index=int(row["chunk_index"]),
            page_number=int(row["page_number"]) if row["page_number"] is not None else None,
            text=str(row["text"]),
            token_count=int(row["token_count"]),
        )

from __future__ import annotations

import uuid

import httpx
import pytest

from app.config import get_settings


@pytest.mark.asyncio
async def test_api_smoke_and_actionable_missing_provider_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "rag.sqlite3"))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_COLLECTION", f"api_test_{uuid.uuid4().hex}")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    from app.main import app

    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                assert (await client.get("/healthz")).json() == {"status": "ok"}
                assert (await client.get("/readyz")).json() == {"status": "ready"}
                status = await client.get("/api/v1/status")
                assert status.status_code == 200
                assert status.json()["embedding_configured"] is False
                assert (await client.get("/")).status_code == 200
                assert (await client.get("/api/v1/documents")).json() == []

                unsupported = await client.post(
                    "/api/v1/documents",
                    files={"file": ("image.png", b"not a document", "image/png")},
                )
                assert unsupported.status_code == 415

                missing_key = await client.post(
                    "/api/v1/documents",
                    files={"file": ("notes.txt", b"A valid internal policy paragraph.", "text/plain")},
                )
                assert missing_key.status_code == 503
                assert "OPENAI_API_KEY" in missing_key.json()["detail"]
    finally:
        get_settings.cache_clear()

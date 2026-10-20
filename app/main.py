from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api.routes import router
from app.config import get_settings
from app.services.container import build_container

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    services = build_container(settings)
    application.state.services = services
    try:
        yield
    finally:
        services.vectors.close()


app = FastAPI(
    title="Atlas RAG",
    description="Internal knowledge search with hybrid retrieval and verified citations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/assets", StaticFiles(directory=str(_WEB_DIR / "assets")), name="assets")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness probe: the API process is running."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> dict[str, str]:
    """Readiness probe: metadata and vector stores can serve requests."""
    services = request.app.state.services
    try:
        await run_in_threadpool(services.lexical.health_check)
        await run_in_threadpool(services.vectors.health_check)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Storage dependencies are not ready.") from exc
    return {"status": "ready"}

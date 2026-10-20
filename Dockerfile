FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/atlas

ARG INSTALL_LOCAL_MODELS=false
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then pip install 'sentence-transformers>=3.3,<6.0'; fi

RUN useradd --system --uid 10001 --create-home atlas && mkdir -p /data && chown -R atlas:atlas /data /srv/atlas
COPY --chown=atlas:atlas app ./app
COPY --chown=atlas:atlas web ./web

ENV SQLITE_PATH=/data/rag.sqlite3 \
    QDRANT_PATH=/data/qdrant \
    MAX_UPLOAD_MB=25

USER atlas
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

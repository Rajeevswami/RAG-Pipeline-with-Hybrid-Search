# Atlas — RAG Pipeline with Hybrid Search

A production-minded retrieval-augmented generation service for company knowledge. Index internal PDFs, Word documents, Markdown, text, and HTML; retrieve evidence with **dense vectors + BM25**, fuse and rerank the results, then return a grounded answer with citations checked against the original text.

The repository includes a FastAPI backend, a compact browser UI, embedded-Qdrant local mode, Docker Compose with a Qdrant service, and tests for parsing, ingestion, chunking, hybrid ranking, persistence, evaluation metrics, and citation checks.

## Why this is more than a PDF chatbot

- **Hybrid retrieval:** OpenAI or local sentence-transformer embeddings in Qdrant, plus SQLite FTS5's BM25 search. Reciprocal-rank fusion (RRF) combines both result lists so semantic matches and exact policy terms both count.
- **Chunking with provenance:** configurable, overlapping chunks prefer paragraph/sentence boundaries and stay within PDF pages. Citations include the original filename and PDF page where available.
- **Reranking:** a lightweight deterministic overlap reranker runs by default. An optional sentence-transformers CrossEncoder can be enabled for learned reranking.
- **Grounded answers:** GPT-4o-family and Claude Sonnet-family chat models are supported. The prompt treats retrieved documents as untrusted evidence, not instructions, and asks the model to abstain when the evidence is insufficient.
- **Citation verification:** every factual sentence must reference a retrieved source; every cited source ID must belong to the current retrieval; and each returned support quote must be an exact contiguous excerpt from that source (whitespace-normalized). If the model's citations fail those checks, Atlas falls back to a matching verbatim source sentence or abstains.
- **Operations:** Dockerized API, persistent Qdrant and SQLite volumes, upload size/type checks, duplicate detection, health/status endpoints, and no document text in application logs.

> Citation validation proves provenance and quote integrity; it is not a formal entailment or truthfulness proof. Review high-impact answers and evaluate the pipeline against your own documents before production use.

## Quick start (Docker)

Requirements: Docker Engine and Docker Compose.

```bash
cp .env.example .env
```

Add an `OPENAI_API_KEY` to `.env`. By default OpenAI is used for both embeddings and answers. To use Claude for answers, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`; OpenAI is still the default embedding provider unless you switch to local embeddings.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). Upload a few internal documents, then ask a question. API reference: [http://localhost:8000/docs](http://localhost:8000/docs).

Docker Compose starts a Qdrant server and persists data in named volumes (`qdrant_data` and `atlas_data`). To stop the containers without removing indexed data:

```bash
docker compose down
```

To permanently remove those local indexes as well, use `docker compose down -v`.

## Run locally without Docker

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY to .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

With `QDRANT_URL` blank, the Python client runs persistent embedded Qdrant under `data/qdrant`; SQLite/FTS data is stored at `data/rag.sqlite3`. Embedded Qdrant is intended for local development and a **single API process**. Use the Qdrant server (as in Compose) if you need multiple API workers or a shared vector store.

### Optional provider/model setup

| Goal | Configuration |
| --- | --- |
| GPT-4o-family answers | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `OPENAI_CHAT_MODEL=gpt-4o` (default is the smaller `gpt-4o-mini`) |
| Claude Sonnet answers | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...`, set `ANTHROPIC_MODEL` as desired |
| Local dense embeddings | `pip install -r requirements-local.txt`, then set `EMBEDDING_PROVIDER=sentence-transformers`. With Compose, also set `INSTALL_LOCAL_MODELS=true` and rebuild the API image. |
| Learned CrossEncoder reranker | `pip install -r requirements-reranker.txt`, then set `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`. With Compose, also set `INSTALL_LOCAL_MODELS=true` and rebuild the API image. |

Local sentence-transformer models are downloaded/cached by the library on first use. The local provider uses a conservative 160/32 chunk/overlap default to avoid silently truncating passages in short-context sentence encoders; tune `LOCAL_EMBEDDING_CHUNK_SIZE_TOKENS` and `LOCAL_EMBEDDING_CHUNK_OVERLAP_TOKENS` for your model. If changing embedding provider, model, or vector dimension on an existing index, create a new collection and re-index your documents; vectors from different embedding spaces must not be mixed.

## Configuration

All settings can be set in `.env` or as environment variables. See [`.env.example`](.env.example) for the complete list.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` answer generation |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic chat model |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `sentence-transformers` dense vectors |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Embedding vector size; must match the Qdrant collection |
| `LOCAL_EMBEDDING_CHUNK_SIZE_TOKENS` / `LOCAL_EMBEDDING_CHUNK_OVERLAP_TOKENS` | `160` / `32` | Conservative chunking defaults for local sentence-transformer context windows |
| `QDRANT_URL` | blank | Blank means embedded local Qdrant; set a URL for a Qdrant server |
| `QDRANT_COLLECTION` | `company_docs` | Collection for dense vectors |
| `SQLITE_PATH` | `data/rag.sqlite3` | Documents, chunks, and FTS5 index |
| `CHUNK_SIZE_TOKENS` | `420` | Approximate whitespace-token budget per chunk |
| `CHUNK_OVERLAP_TOKENS` | `64` | Approximate overlap between chunks on the same page |
| `RETRIEVAL_CANDIDATES` | `30` | Candidate count from each retriever before fusion/reranking |
| `DEFAULT_TOP_K` | `6` | Evidence passages supplied to the answer model |
| `MAX_TOP_K` | `10` | API limit for per-query evidence |
| `DENSE_WEIGHT` / `LEXICAL_WEIGHT` | `0.55` / `0.45` | Relative RRF contributions |
| `MAX_UPLOAD_MB` | `25` | Maximum upload size |

The status endpoint reports whether the selected answer and embedding providers are configured. The service can start without API keys, but OpenAI-backed indexing and/or answers will return an actionable configuration error until the needed key is supplied.

## Retrieval and citation flow

1. **Parse** — extract page-by-page text from PDFs, paragraphs/tables from DOCX, and readable text from HTML or plain-text files. Scanned PDFs without selectable text are rejected with an OCR hint; OCR is not bundled.
2. **Chunk** — split within page boundaries, preferring nearby paragraph or sentence ends. Chunks overlap to reduce boundary misses; token counts are approximate and intentionally avoid requiring a tokenizer download.
3. **Index** — store chunk metadata/text and a BM25 FTS5 index in SQLite; embed chunks and upsert vectors with document metadata in Qdrant. Duplicate file bytes are rejected by SHA-256 checksum.
4. **Retrieve** — get dense and BM25 result lists, fuse ranks with weighted RRF, then rerank the combined candidates. The default reranker uses query-term overlap plus fusion score; configure a CrossEncoder for learned reranking.
5. **Answer and verify** — send the question and a compact set of labeled evidence snippets to the chosen LLM. The response must include inline `[S1]`-style references and exact support quotes. Atlas checks source membership, sentence-level markers, and exact quote matches before returning citations.

## API

- `GET /healthz` — process liveness.
- `GET /readyz` — metadata and vector-store readiness (used by the container health check).
- `GET /api/v1/status` — storage health, provider configuration, models, and indexed-document count.
- `GET /api/v1/documents` — list indexed documents.
- `POST /api/v1/documents` — upload one multipart `file` (`.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.htm`).
- `DELETE /api/v1/documents/{document_id}` — remove a document from both indexes.
- `POST /api/v1/query` — hybrid search and grounded answer.

Example:

```bash
curl -F 'file=@./employee-handbook.pdf' http://localhost:8000/api/v1/documents

curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many days of PTO can be carried over?","top_k":5}'
```

A response includes `answer`, `abstained`, `citation_verified`, `verification_method`, `citations`, and the retrieved `sources`. `verification_method` is `quote_match`, `extractive_fallback`, or `abstained`.

## Retrieval evaluation

The evaluation CLI measures retrieval quality without making any LLM calls. Create a JSONL relevance set using document IDs from `GET /api/v1/documents` (or chunk IDs from `sources`/`citations` in a query response if you maintain chunk-level labels):

```json
{"question":"How do employees request PTO?","relevant_document_ids":["<document-id>"]}
```

Run it from the repository root with the same embedding/Qdrant configuration used by the app:

```bash
python -m scripts.evaluate_retrieval ./evaluation-set.jsonl --k 6
```

It reports mean Recall@k, Precision@k, MRR, and nDCG, plus per-query metrics. Use a held-out set of real company questions and labeled evidence before tuning chunk sizes, retrieval weights, or rerankers. If using embedded local Qdrant, stop the API before running the evaluator; use a Qdrant server (`QDRANT_URL`) to evaluate concurrently.

## Development and tests

```bash
pip install -r requirements-dev.txt
pytest
```

Optional formatting/lint check:

```bash
ruff check app scripts tests
```

## Security and production notes

This is a strong starting point, not a complete enterprise security boundary. The sample UI/API do not implement user authentication, tenant isolation, document ACL enforcement, OCR, audit retention, or deletion in provider-side backups. Put the service behind your organization’s SSO/reverse proxy, restrict network access, define document access policy before indexing, and review provider retention/data-processing terms. Do not expose an unauthenticated instance containing private company documents to the public internet.

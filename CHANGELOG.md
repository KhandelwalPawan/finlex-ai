# Changelog

All notable changes to **FinLex AI** are documented here.

---

## [Unreleased] — 2026-09-05

### Added

- **Conversation memory** (`ConversationMemory`): rolling buffer of last 5 turns injected into each prompt so follow-up questions work naturally.
- **Streaming responses**: `ProductionRAGChain.invoke_streaming()` yields LLM tokens via `chain.stream()`; Streamlit UI renders them token-by-token with `st.write_stream`.
- **Request timeout enforcement**: `_run_with_timeout()` wraps LLM calls with `concurrent.futures.ThreadPoolExecutor`; exceeding `RAG_REQUEST_TIMEOUT_SECONDS` returns a clear `TimeoutError` mapped to HTTP 504.
- **Input length cap**: questions longer than 500 characters are rejected before any retrieval.
- **API CORS middleware** via `starlette.middleware.cors.CORSMiddleware`.
- **API rate limiting**: simple in-memory token bucket — 20 requests per 60 s per IP, returns HTTP 429 with `Retry-After` header.
- **`/sources` API endpoint**: lists indexed documents with chunk and page counts.
- **`/chat/stream` SSE endpoint**: Server-Sent Events streaming for API clients.
- **PDF upload widget** in Streamlit sidebar: saves files to `data/` and optionally triggers re-ingestion via subprocess.
- **Clear chat button** in sidebar; also resets `ConversationMemory`.
- **Confidence color badges**: high=green, medium=amber, low=red, none=grey via inline CSS.
- **Streamlit dark theme** with green primary color and `maxUploadSize = 200 MB`.
- **`docker-compose.yml`**: `finlex-ui` (Streamlit:8501) and `finlex-api` (uvicorn:8000) with bind-mounted volumes, env_file, health checks, and restart policies.
- **`tests/test_api.py`**: 10 unit tests covering all API endpoints with mocked chain, healthcheck, rate limits (429), and streaming SSE.
- **`tests/test_healthcheck.py`**: 5 unit tests for `run_healthcheck()` edge cases.
- **`tests/test_ingest.py`**: 5 unit tests for `_pdf_files()` and `_load_and_split()` with mocked loader.
- **`tests/test_rag_pipeline.py`**: 11 unit tests covering memory, streaming guards, length limit, and security guardrails.
- **`generate_docs.py`**: PDF generator using PyMuPDF for regulatory compendia (DPDP 2023, IBC 2016, FEMA 1999).
- **`test_query_runner.py`**: Automated end-to-end live test runner executing questions across documents and guardrails.
- **Expanded `eval_questions.json`**: 10 questions covering inventory, live-data refusal, prompt injection, empty input, domain-grounded (ICA, income-tax, SEBI, RBI), out-of-scope, and cross-document synthesis.

### Changed

- **`.env`**: now populated with all configuration variables from `.env.example`.
- **`Dockerfile`**: non-root user (`finlex`), both ports exposed (8501 + 8000), proper `HEALTHCHECK` flags.
- **`app.py`**: sidebar reformatted (settings as key/value list instead of raw JSON), `max_chars` on chat input, streaming render path.
- **`api.py`**: fully restructured with shared `_parse_question()` helper and all new endpoints.

---

## Phase 6 — Evaluation and Deployment

- Added `evaluate.py` regression runner and `eval_questions.json`.
- Added `Dockerfile` and `.dockerignore`.

## Phase 5 — App and API Readiness

- Streamlit UI: health status, citations, runtime settings.
- Starlette API: `/health` and `/chat` endpoints.

## Phase 4 — Ingestion Reliability

- Validated PDF input, preserved metadata and checksums, atomic vectorstore rebuild.

## Phase 3 — Grounded RAG Quality

- Relevance score filtering, untrusted-evidence prompt, refusal on weak context, structured citations.

## Phase 2 — Secure Vectorstore Handling

- FAISS index validated against `manifest.json` with SHA-256 checksums.

## Phase 1 — Runtime Configuration

- All settings behind environment variables with `config.py` and `.env.example`.

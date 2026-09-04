# FinLex AI

FinLex AI is a production-grade RAG-based Q&A assistant for finance and legal PDFs. It ingests trusted documents, builds a FAISS vector index, retrieves relevant evidence, and answers with grounded citations — refusing or flagging when evidence is insufficient.

## Features

- 📄 **Multi-document RAG** — indexes PDFs (finance acts, tax law, regulatory docs, reports)
- 💬 **Conversation memory** — follow-up questions use the last 5 turns as context
- ⚡ **Streaming responses** — token-by-token output in the UI and via SSE API
- 🔐 **Prompt injection defense** — unsafe queries refused before retrieval
- 📎 **Grounded citations** — every answer shows source, page, relevance score, and excerpt
- 🔒 **Vectorstore integrity** — SHA-256 checksums validated on every load
- 🌐 **REST API** — CORS-enabled, rate-limited, with streaming SSE endpoint
- 🐳 **Docker-ready** — Dockerfile + docker-compose for UI and API services
- 📤 **PDF upload** — add documents via the Streamlit sidebar and re-ingest in one click

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama + Llama 3.2 (local, default) |
| Embeddings | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| RAG framework | LangChain |
| PDF loading | PyMuPDF |
| UI | Streamlit |
| API | Starlette + uvicorn |
| Testing | unittest + starlette TestClient |

## Setup

```powershell
python -m venv rag-venv
.\rag-venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env if you want to change models or paths
```

Install Ollama and pull the default local model:

```powershell
ollama pull llama3.2
```

Add PDF documents to `data/`, then build the vectorstore:

```powershell
python ingest.py
```

## Run the Chat UI

```powershell
streamlit run app.py
```

- Upload PDFs directly from the sidebar and click **Re-ingest documents** to rebuild the index.
- Click **🗑️ Clear chat** to reset the conversation and memory.
- Answers stream token-by-token with color-coded evidence confidence badges.

> **Note:** If you rebuild `vectorstore/` while Streamlit is open, restart Streamlit before refreshing the browser. Windows can briefly deny access to replaced FAISS files in a running app process.

## Run the API

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health and config |
| `GET` | `/sources` | List indexed documents |
| `POST` | `/chat` | Full answer with citations |
| `POST` | `/chat/stream` | Server-Sent Events streaming |

**Rate limit:** 20 requests per minute per IP. Exceeding returns HTTP 429.

#### `/chat` request body
```json
{ "question": "What does the Income Tax Act say about deductions?" }
```

#### `/chat/stream` SSE format
```
data: {"token": "The"}
data: {"token": " Income"}
...
data: [DONE]
```

## Run with Docker Compose

```powershell
docker compose up --build
```

- Streamlit UI: http://localhost:8501
- API: http://localhost:8000

Both services share `data/` and `vectorstore/` via bind mounts. Run `python ingest.py` once before starting containers.

## Run Health Checks and Tests

```powershell
python healthcheck.py
python -m pytest tests/ -v
```

## Run Evaluation

```powershell
python evaluate.py
```

Runs 10 evaluation questions covering: document inventory, live-data refusal, prompt injection, grounded legal/tax/regulatory answers, out-of-scope refusal, and cross-document synthesis. Results written to `eval_results.json`.

## Configuration

All settings are read from environment variables. Copy `.env.example` to `.env`:

| Variable | Default | Description |
|---|---|---|
| `RAG_LLM_MODEL` | `llama3.2` | Ollama model name |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embeddings |
| `RAG_CHUNK_SIZE` | `700` | Characters per chunk |
| `RAG_RETRIEVAL_K` | `6` | Chunks retrieved per query |
| `RAG_MIN_RELEVANCE_SCORE` | `0.2` | Minimum cosine similarity |
| `RAG_REQUEST_TIMEOUT_SECONDS` | `120` | LLM timeout |
| `RAG_REQUIRE_TRUSTED_VECTORSTORE` | `true` | Enforce manifest/checksum |
| `HF_TOKEN` | — | HuggingFace token (for gated models) |

## Production Notes

- Responses are grounded in indexed documents only. Do not present as legal, tax, investment, or compliance advice without qualified human review.
- The default `RAG_LLM_NUM_GPU=0` favors CPU stability. Increase after validating Ollama + GPU driver compatibility.
- The Docker image runs as a non-root user (`finlex`) for security.
- The API rate limiter is in-memory; use a Redis-backed limiter (e.g., `slowapi`) for multi-process deployments.

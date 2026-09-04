from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import AsyncGenerator

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from healthcheck import run_healthcheck
from rag_pipeline import get_rag_chain, MAX_QUESTION_LENGTH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter: max 20 req/minute per IP
# ---------------------------------------------------------------------------
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 20

_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate limit exceeded."""
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]
    # Evict timestamps older than the window
    _rate_buckets[client_ip] = [t for t in bucket if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_buckets[client_ip]) >= _RATE_LIMIT_MAX:
        return False
    _rate_buckets[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Lazy chain singleton
# ---------------------------------------------------------------------------
_chain = None


def chain():
    global _chain
    if _chain is None:
        _chain = get_rag_chain()
    return _chain


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------
def _rate_limit_response() -> JSONResponse:
    return JSONResponse(
        {"error": "Rate limit exceeded. Max 20 requests per minute."},
        status_code=429,
        headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
    )


async def _parse_question(request: Request) -> tuple[str | None, JSONResponse | None]:
    """Parse and validate question from request body. Returns (question, error_response)."""
    try:
        payload = await request.json()
    except Exception:
        return None, JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, JSONResponse({"error": "`question` is required and must be a non-empty string."}, status_code=400)

    if len(question) > MAX_QUESTION_LENGTH:
        return None, JSONResponse(
            {"error": f"`question` must be {MAX_QUESTION_LENGTH} characters or fewer."},
            status_code=422,
        )
    return question, None


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------
async def health(_: Request) -> JSONResponse:
    status = run_healthcheck()
    http_status = 200 if status["status"] == "ok" else 503
    return JSONResponse(status, status_code=http_status)


async def sources(request: Request) -> JSONResponse:
    ip = _client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limit_response()
    try:
        inventory = chain().source_inventory()
        return JSONResponse({"sources": inventory})
    except Exception:
        LOGGER.exception("sources_request_failed")
        return JSONResponse({"error": "Failed to retrieve source inventory."}, status_code=500)


async def chat(request: Request) -> JSONResponse:
    ip = _client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limit_response()

    question, err = await _parse_question(request)
    if err is not None:
        return err

    try:
        return JSONResponse(chain().invoke(question))
    except TimeoutError as exc:
        return JSONResponse({"error": str(exc)}, status_code=504)
    except Exception as exc:
        LOGGER.exception("chat_request_failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def chat_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events streaming endpoint."""
    ip = _client_ip(request)
    if not _check_rate_limit(ip):
        return _rate_limit_response()

    question, err = await _parse_question(request)
    if err is not None:
        return err

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            for token in chain().invoke_streaming(question):
                # SSE format: data: <payload>\n\n
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except TimeoutError as exc:
            error_payload = json.dumps({"error": str(exc)})
            yield f"data: {error_payload}\n\n"
        except Exception as exc:
            LOGGER.exception("chat_stream_failed")
            error_payload = json.dumps({"error": str(exc)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/sources", sources, methods=["GET"]),
        Route("/chat", chat, methods=["POST"]),
        Route("/chat/stream", chat_stream, methods=["POST"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
        )
    ],
)

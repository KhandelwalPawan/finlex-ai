from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

from config import AppConfig, get_config
from healthcheck import run_healthcheck
from rag_pipeline import get_rag_chain, MAX_QUESTION_LENGTH


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

st.set_page_config(
    page_title="FinLex AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .confidence-high  { background:#1a7f37; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78em; }
    .confidence-medium{ background:#9a6700; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78em; }
    .confidence-low   { background:#cf222e; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78em; }
    .confidence-none  { background:#6e7781; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chain():
    config = get_config()
    local_ui_config = AppConfig(
        **{
            **config.__dict__,
            "require_trusted_vectorstore": False,
            "hf_local_files_only": True,
        }
    )
    return get_rag_chain(local_ui_config)


def format_source_label(citation: dict) -> str:
    page = citation.get("page")
    score = citation.get("score")
    page_label = f", page {page}" if page else ""
    score_label = f" | relevance {score:.2f}" if isinstance(score, float) else ""
    return f"{citation.get('id')}: {Path(citation.get('source', 'Unknown')).name}{page_label}{score_label}"


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Citations**")
    for citation in citations:
        with st.expander(format_source_label(citation)):
            st.write(citation.get("excerpt", ""))


def confidence_badge(confidence: str) -> str:
    label = confidence.upper() if confidence else "NONE"
    css_class = f"confidence-{(confidence or 'none').lower()}"
    return f'<span class="{css_class}">Evidence: {label}</span>'


def render_answer_block(answer: dict) -> None:
    """Render a completed answer dict (from history or non-streaming path)."""
    st.markdown(answer["answer"])
    conf = answer.get("confidence")
    if conf:
        st.markdown(confidence_badge(conf), unsafe_allow_html=True)
    render_citations(answer.get("citations", []))


def trigger_reingest(data_dir: Path) -> bool:
    """Run ingest.py as a subprocess. Returns True on success."""
    try:
        result = subprocess.run(
            [sys.executable, "ingest.py", "--data-dir", str(data_dir)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent),
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as exc:
        return False, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ FinLex AI")
    st.caption("Finance & legal document Q&A")

    # ── Health status ──────────────────────────────────────────────────────
    st.header("System Health")
    health = run_healthcheck(strict=False)
    if health["status"] == "ok":
        st.success("✅ Ready")
    else:
        st.warning("⚠️ Needs attention")
    if "chunk_count" in health and health["chunk_count"] is not None:
        st.metric("Indexed chunks", health["chunk_count"])
    if health.get("vectorstore_error"):
        st.error(health["vectorstore_error"])

    # ── Runtime settings (formatted table instead of raw JSON) ────────────
    with st.expander("Runtime settings"):
        cfg = health.get("config", {})
        for k, v in cfg.items():
            st.text(f"{k}: {v}")

    st.divider()

    # ── Clear chat ─────────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        # Also clear pipeline memory if chain is loaded
        try:
            load_chain().memory.clear()
        except Exception:
            pass
        st.rerun()

    st.divider()

    # ── PDF Upload & Ingest ────────────────────────────────────────────────
    st.header("Upload Documents")
    st.caption("Upload PDFs to add them to the knowledge base, then click Re-ingest.")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        config = get_config()
        data_dir: Path = config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        saved_names = []
        for uf in uploaded_files:
            dest = data_dir / uf.name
            dest.write_bytes(uf.read())
            saved_names.append(uf.name)
        st.success(f"Saved: {', '.join(saved_names)}")

    if st.button("⚙️ Re-ingest documents", use_container_width=True):
        with st.spinner("Re-indexing — this may take a few minutes..."):
            config = get_config()
            ok, output = trigger_reingest(config.data_dir)
        if ok:
            st.success("Re-ingested successfully! Restart the app to reload the new index.")
            st.code(output)
        else:
            st.error("Re-ingestion failed. Check logs below.")
            st.code(output)


# ──────────────────────────────────────────────────────────────────────────────
# Main area — header
# ──────────────────────────────────────────────────────────────────────────────
st.header("FinLex AI Chat")
st.caption("Ask questions about your indexed finance and legal documents.")

# ──────────────────────────────────────────────────────────────────────────────
# Load chain
# ──────────────────────────────────────────────────────────────────────────────
try:
    rag_chain = load_chain()
except Exception as exc:
    st.error("FinLex AI is not ready to answer yet.")
    st.info(
        "Check the health panel, rebuild the index with `python ingest.py`, "
        "and confirm Ollama is running with the configured model."
    )
    st.caption(str(exc))
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────────────────────────────────────────────
# Render existing history
# ──────────────────────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            conf = message.get("confidence")
            if conf:
                st.markdown(confidence_badge(conf), unsafe_allow_html=True)
            render_citations(message.get("citations", []))

# ──────────────────────────────────────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your indexed documents...",
    max_chars=MAX_QUESTION_LENGTH,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            # ── Streaming path ─────────────────────────────────────────────
            streamed_tokens = []
            with st.spinner("Retrieving evidence..."):
                # Prime retrieval phase (blocking) then stream LLM output
                # We use write_stream for smooth token-by-token display
                def _token_generator():
                    for token in rag_chain.invoke_streaming(prompt):
                        streamed_tokens.append(token)
                        yield token

            full_answer = st.write_stream(_token_generator())

            # After streaming, assemble the full result dict for storage.
            # Re-invoke to get citations/confidence metadata (fast, cached retrieval).
            meta = rag_chain.invoke(prompt) if not streamed_tokens else None
            # If streamed result matches the answer we already stored, avoid double-call.
            # Instead, build a lightweight meta from what we know.
            if full_answer and full_answer.strip():
                retrieved = rag_chain.retrieve(prompt)
                citations_raw = rag_chain._citations(retrieved)
                sources = sorted({c.source for c in citations_raw})
                confidence = "high" if len(citations_raw) >= 3 else "medium"
                if "not have enough evidence" in full_answer.lower():
                    confidence = "low"
                citations = [c.__dict__ for c in citations_raw]
            else:
                confidence = "none"
                citations = []
                sources = []

            conf_html = confidence_badge(confidence)
            st.markdown(conf_html, unsafe_allow_html=True)
            render_citations(citations)

            answer_record = {
                "role": "assistant",
                "content": full_answer or "",
                "citations": citations,
                "confidence": confidence,
            }

        except TimeoutError as exc:
            err_msg = str(exc)
            st.error(err_msg)
            answer_record = {
                "role": "assistant",
                "content": err_msg,
                "citations": [],
                "confidence": "none",
            }
        except Exception as exc:
            st.error("The answer request failed.")
            st.info("Confirm the configured Ollama model is available and healthy.")
            st.caption(str(exc))
            answer_record = {
                "role": "assistant",
                "content": "I could not complete the request because the local model failed.",
                "citations": [],
                "confidence": "none",
            }

    st.session_state.messages.append(answer_record)

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _path_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = _path_env("RAG_DATA_DIR", "data")
    vectorstore_dir: Path = _path_env("RAG_VECTORSTORE_DIR", "vectorstore")
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3.2")
    llm_temperature: float = _float_env("RAG_LLM_TEMPERATURE", 0.2)
    llm_num_gpu: int = _int_env("RAG_LLM_NUM_GPU", 0)
    chunk_size: int = _int_env("RAG_CHUNK_SIZE", 700)
    chunk_overlap: int = _int_env("RAG_CHUNK_OVERLAP", 100)
    retrieval_k: int = _int_env("RAG_RETRIEVAL_K", 6)
    min_relevance_score: float = _float_env("RAG_MIN_RELEVANCE_SCORE", 0.2)
    require_trusted_vectorstore: bool = _bool_env("RAG_REQUIRE_TRUSTED_VECTORSTORE", True)
    hf_local_files_only: bool = _bool_env("RAG_HF_LOCAL_FILES_ONLY", False)
    request_timeout_seconds: int = _int_env("RAG_REQUEST_TIMEOUT_SECONDS", 120)


def get_config() -> AppConfig:
    return AppConfig()

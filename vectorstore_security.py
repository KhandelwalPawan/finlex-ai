from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MANIFEST_NAME = "manifest.json"
INDEX_FILES = ("index.faiss", "index.pkl")


class VectorstoreTrustError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VectorstoreTrustError(
            "Vectorstore manifest is missing. Rebuild it with `python ingest.py` "
            "or set RAG_REQUIRE_TRUSTED_VECTORSTORE=false only for trusted local demos."
        ) from exc
    except PermissionError as exc:
        raise VectorstoreTrustError(
            f"Cannot read vectorstore manifest due to permissions: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise VectorstoreTrustError(f"Vectorstore manifest is not valid JSON: {path}") from exc


def build_manifest(
    vectorstore_dir: Path,
    source_files: Iterable[Path],
    embedding_model: str,
    chunk_count: int,
) -> dict:
    index_files = {
        name: sha256_file(vectorstore_dir / name)
        for name in INDEX_FILES
        if (vectorstore_dir / name).exists()
    }
    sources = [
        {
            "name": file.name,
            "path": str(file),
            "sha256": sha256_file(file),
        }
        for file in source_files
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "chunk_count": chunk_count,
        "index_files": index_files,
        "sources": sources,
    }


def write_manifest(
    vectorstore_dir: Path,
    source_files: Iterable[Path],
    embedding_model: str,
    chunk_count: int,
) -> None:
    manifest = build_manifest(vectorstore_dir, source_files, embedding_model, chunk_count)
    path = vectorstore_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_vectorstore(
    vectorstore_dir: Path,
    require_manifest: bool = True,
    check_index_files: bool = True,
) -> dict | None:
    manifest_path = vectorstore_dir / MANIFEST_NAME
    try:
        manifest = _read_json(manifest_path)
    except VectorstoreTrustError:
        if require_manifest:
            raise
        return None

    if not check_index_files:
        return manifest

    recorded = manifest.get("index_files", {})
    for name in INDEX_FILES:
        path = vectorstore_dir / name
        try:
            actual = sha256_file(path)
        except FileNotFoundError as exc:
            raise VectorstoreTrustError(f"Vectorstore is missing required file: {name}") from exc
        except PermissionError as exc:
            raise VectorstoreTrustError(
                f"Cannot read vectorstore file due to permissions: {path}"
            ) from exc
        expected = recorded.get(name)
        if expected != actual:
            raise VectorstoreTrustError(
                f"Vectorstore checksum mismatch for {name}. Rebuild with `python ingest.py`."
            )
    return manifest

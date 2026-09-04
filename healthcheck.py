from __future__ import annotations

from dataclasses import asdict

from config import AppConfig, get_config
from vectorstore_security import VectorstoreTrustError, validate_vectorstore


def run_healthcheck(config: AppConfig | None = None, strict: bool = True) -> dict:
    config = config or get_config()
    checks = {
        "data_dir_exists": config.data_dir.exists(),
        "vectorstore_dir_exists": config.vectorstore_dir.exists(),
        "vectorstore_trusted": False,
        "config": {
            key: str(value) if key.endswith("_dir") else value
            for key, value in asdict(config).items()
            if key not in {"request_timeout_seconds"}
        },
    }

    try:
        manifest = validate_vectorstore(
            config.vectorstore_dir,
            require_manifest=config.require_trusted_vectorstore and strict,
            check_index_files=strict,
        )
        checks["vectorstore_trusted"] = manifest is not None or not strict
        checks["chunk_count"] = manifest.get("chunk_count") if manifest else None
    except VectorstoreTrustError as exc:
        checks["vectorstore_error"] = str(exc)
    except OSError as exc:
        checks["vectorstore_error"] = f"Cannot inspect vectorstore: {exc}"

    checks["status"] = "ok" if checks["data_dir_exists"] and checks["vectorstore_trusted"] else "degraded"
    return checks


if __name__ == "__main__":
    import json

    print(json.dumps(run_healthcheck(), indent=2))

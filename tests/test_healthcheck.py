from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthcheck import run_healthcheck
from config import AppConfig


def _make_config(data_dir: Path, vectorstore_dir: Path) -> AppConfig:
    return AppConfig(
        data_dir=data_dir,
        vectorstore_dir=vectorstore_dir,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="llama3.2",
        llm_temperature=0.2,
        llm_num_gpu=0,
        chunk_size=700,
        chunk_overlap=100,
        retrieval_k=6,
        min_relevance_score=0.2,
        require_trusted_vectorstore=False,
        hf_local_files_only=False,
        request_timeout_seconds=30,
    )


class HealthcheckTest(unittest.TestCase):
    def test_missing_data_dir_is_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(
                data_dir=Path(tmp) / "nonexistent",
                vectorstore_dir=Path(tmp) / "vs",
            )
            result = run_healthcheck(config=config, strict=False)
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["data_dir_exists"])

    def test_ok_when_both_dirs_exist_and_not_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            vs_dir = Path(tmp) / "vs"
            vs_dir.mkdir()
            config = _make_config(data_dir=data_dir, vectorstore_dir=vs_dir)
            result = run_healthcheck(config=config, strict=False)
        # In non-strict mode, vectorstore is considered trusted even without manifest
        self.assertTrue(result["data_dir_exists"])
        self.assertEqual(result["status"], "ok")

    def test_config_is_included_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(
                data_dir=Path(tmp),
                vectorstore_dir=Path(tmp) / "vs",
            )
            result = run_healthcheck(config=config, strict=False)
        self.assertIn("config", result)
        self.assertIsInstance(result["config"], dict)

    def test_strict_mode_fails_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            vs_dir = Path(tmp) / "vs"
            vs_dir.mkdir()
            config = _make_config(data_dir=data_dir, vectorstore_dir=vs_dir)
            # require_trusted_vectorstore = False in config, but strict=True in call
            config_strict = AppConfig(
                **{**config.__dict__, "require_trusted_vectorstore": True}
            )
            result = run_healthcheck(config=config_strict, strict=True)
        # Should report vectorstore error (missing manifest)
        self.assertIn("vectorstore_error", result)

    def test_status_key_always_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(
                data_dir=Path(tmp),
                vectorstore_dir=Path(tmp) / "vs",
            )
            result = run_healthcheck(config=config, strict=False)
        self.assertIn("status", result)
        self.assertIn(result["status"], {"ok", "degraded"})


if __name__ == "__main__":
    unittest.main()

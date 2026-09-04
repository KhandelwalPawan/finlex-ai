from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vectorstore_security import (
    VectorstoreTrustError,
    validate_vectorstore,
    write_manifest,
)


class VectorstoreSecurityTest(unittest.TestCase):
    def test_missing_index_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(VectorstoreTrustError):
                validate_vectorstore(Path(directory))

    def test_manifest_checksums_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.faiss").write_text("faiss", encoding="utf-8")
            (root / "index.pkl").write_text("pickle", encoding="utf-8")
            source = root / "source.pdf"
            source.write_text("pdf", encoding="utf-8")
            write_manifest(root, [source], "embedding-model", 1)

            manifest = validate_vectorstore(root)

            self.assertEqual(manifest["chunk_count"], 1)

    def test_changed_index_after_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.faiss").write_text("faiss", encoding="utf-8")
            (root / "index.pkl").write_text("pickle", encoding="utf-8")
            source = root / "source.pdf"
            source.write_text("pdf", encoding="utf-8")
            write_manifest(root, [source], "embedding-model", 1)
            (root / "index.pkl").write_text("changed", encoding="utf-8")

            with self.assertRaises(VectorstoreTrustError):
                validate_vectorstore(root)


if __name__ == "__main__":
    unittest.main()

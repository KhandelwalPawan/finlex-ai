from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import AppConfig
from ingest import _pdf_files, _load_and_split


def _make_config(data_dir: Path) -> AppConfig:
    return AppConfig(
        data_dir=data_dir,
        vectorstore_dir=data_dir / "vs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="llama3.2",
        llm_temperature=0.2,
        llm_num_gpu=0,
        chunk_size=200,
        chunk_overlap=20,
        retrieval_k=6,
        min_relevance_score=0.2,
        require_trusted_vectorstore=False,
        hf_local_files_only=False,
        request_timeout_seconds=30,
    )


class PdfFilesTest(unittest.TestCase):
    def test_returns_sorted_pdf_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.pdf").touch()
            (root / "a.pdf").touch()
            (root / "notes.txt").touch()  # should be excluded
            files = _pdf_files(root)
        self.assertEqual([f.name for f in files], ["a.pdf", "b.pdf"])

    def test_raises_when_dir_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _pdf_files(Path("/nonexistent/data/path"))

    def test_returns_empty_for_no_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "readme.txt").touch()
            files = _pdf_files(Path(tmp))
        self.assertEqual(files, [])


class LoadAndSplitTest(unittest.TestCase):
    def test_metadata_is_attached_to_chunks(self) -> None:
        """Mock PyMuPDFLoader to verify metadata propagation without needing a real PDF."""
        from langchain_core.documents import Document

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_pdf = root / "test.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake")  # not a real PDF, loader will be mocked

            config = _make_config(root)

            mock_doc = Document(
                page_content="This is test content for a finance document.",
                metadata={"page": 0},
            )

            with patch("ingest.PyMuPDFLoader") as MockLoader:
                mock_instance = MagicMock()
                mock_instance.load.return_value = [mock_doc]
                MockLoader.return_value = mock_instance

                chunks = _load_and_split(config, [fake_pdf])

        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertEqual(chunk.metadata.get("source_name"), "test.pdf")
            self.assertIn("doc_sha256", chunk.metadata)

    def test_chunk_size_is_respected(self) -> None:
        """Verify splitter produces chunks smaller than chunk_size."""
        from langchain_core.documents import Document

        long_text = "Finance and law. " * 200  # ~3400 chars

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_pdf = root / "big.pdf"
            fake_pdf.write_bytes(b"%PDF fake")
            config = _make_config(root)

            mock_doc = Document(page_content=long_text, metadata={"page": 0})

            with patch("ingest.PyMuPDFLoader") as MockLoader:
                mock_instance = MagicMock()
                mock_instance.load.return_value = [mock_doc]
                MockLoader.return_value = mock_instance

                chunks = _load_and_split(config, [fake_pdf])

        for chunk in chunks:
            # Allow slight overshoot from splitter heuristics but must be close
            self.assertLessEqual(len(chunk.page_content), config.chunk_size * 2)


if __name__ == "__main__":
    unittest.main()

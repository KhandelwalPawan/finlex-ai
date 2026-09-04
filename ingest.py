from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppConfig, get_config
from vectorstore_security import sha256_file, write_manifest


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _pdf_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    return sorted(data_dir.glob("*.pdf"))


def _load_and_split(config: AppConfig, files: list[Path]):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )
    chunks = []
    for file in files:
        LOGGER.info("loading %s", file.name)
        loader = PyMuPDFLoader(str(file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(file)
            doc.metadata["source_name"] = file.name
            doc.metadata["doc_sha256"] = sha256_file(file)
        chunks.extend(splitter.split_documents(docs))
    return chunks


def build_vectorstore(config: AppConfig) -> int:
    files = _pdf_files(config.data_dir)
    if not files:
        raise ValueError(f"No PDF files found in {config.data_dir}")

    chunks = _load_and_split(config, files)
    if not chunks:
        raise ValueError("No text chunks were extracted from the provided PDFs")

    LOGGER.info("embedding %s chunks with %s", len(chunks), config.embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"local_files_only": config.hf_local_files_only},
    )
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    parent = config.vectorstore_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix="vectorstore-", dir=str(parent)))
    try:
        vectorstore.save_local(str(temp_path))
        write_manifest(temp_path, files, config.embedding_model, len(chunks))
        if config.vectorstore_dir.exists():
            shutil.rmtree(config.vectorstore_dir)
        shutil.move(str(temp_path), str(config.vectorstore_dir))
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)

    LOGGER.info("saved vectorstore to %s", config.vectorstore_dir)
    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FinLex AI vectorstore.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing PDF files.")
    parser.add_argument("--vectorstore-dir", type=Path, help="Output directory for FAISS files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    if args.data_dir:
        config = AppConfig(**{**config.__dict__, "data_dir": args.data_dir})
    if args.vectorstore_dir:
        config = AppConfig(**{**config.__dict__, "vectorstore_dir": args.vectorstore_dir})
    chunk_count = build_vectorstore(config)
    print(f"Indexed {chunk_count} chunks from {config.data_dir}")


if __name__ == "__main__":
    main()

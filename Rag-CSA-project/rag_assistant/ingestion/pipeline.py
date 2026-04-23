"""
Ingestion Pipeline
------------------
Orchestrates the full PDF → ChromaDB pipeline:
  1. Load PDF pages
  2. Chunk pages
  3. Embed chunks
  4. Store in ChromaDB

This is the only entry point external code needs for ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from rag_assistant.ingestion.chunker import chunk_documents
from rag_assistant.ingestion.embedder import get_embedding_model
from rag_assistant.ingestion.loader import load_pdfs
from rag_assistant.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_pdfs(pdf_paths: List[str | Path]) -> VectorStore:
    """
    Run the full ingestion pipeline for one or more PDFs.

    Args:
        pdf_paths: Paths to PDF files to ingest.

    Returns:
        A ready-to-query VectorStore instance.
    """
    logger.info("=== Ingestion Pipeline Start ===")

    # Step 1: Load
    raw_docs = load_pdfs(pdf_paths)
    if not raw_docs:
        raise RuntimeError("No documents were loaded. Check your PDF paths.")

    # Step 2: Chunk
    chunks = chunk_documents(raw_docs)
    if not chunks:
        raise RuntimeError("Chunking produced no output. Check chunk_size config.")

    # Step 3 & 4: Embed + Store
    embedding_model = get_embedding_model()
    store = VectorStore(embedding_model=embedding_model)
    store.add_chunks(chunks)

    logger.info(
        "=== Ingestion Pipeline Complete: %d chunks in store ===",
        store.count(),
    )
    return store


def get_vector_store() -> VectorStore:
    """
    Return a VectorStore connected to the existing ChromaDB collection.
    Use this when the knowledge base has already been ingested.
    """
    embedding_model = get_embedding_model()
    return VectorStore(embedding_model=embedding_model)

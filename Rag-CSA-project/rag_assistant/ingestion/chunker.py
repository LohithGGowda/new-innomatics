"""
Semantic Chunker
----------------
Splits RawDocument pages into overlapping text chunks suitable for embedding.

Strategy: character-level sliding window with configurable size and overlap.
This is a pragmatic choice — semantic sentence-boundary splitting requires an
additional NLP model and adds latency; character chunking with overlap preserves
enough cross-sentence context for most support KB use cases.

Each output Chunk carries forward the source metadata so retrieval results can
always be traced back to the originating PDF page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from rag_assistant.config import config
from rag_assistant.ingestion.loader import RawDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk ready for embedding."""

    chunk_id: str          # unique identifier: "<source_stem>_p<page>_c<idx>"
    text: str              # chunk text content
    source: str            # originating PDF path
    page_number: int       # originating page number
    chunk_index: int       # position within the page's chunks
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split *text* into overlapping windows of *chunk_size* characters.

    Args:
        text: Input string.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of text strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def chunk_documents(
    documents: List[RawDocument],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[Chunk]:
    """
    Convert a list of RawDocuments into a flat list of Chunks.

    Args:
        documents: Pages loaded from PDF(s).
        chunk_size: Override config.chunk_size.
        overlap: Override config.chunk_overlap.

    Returns:
        Flat list of Chunk objects.
    """
    c_size = chunk_size or config.chunk_size
    c_overlap = overlap or config.chunk_overlap

    all_chunks: List[Chunk] = []

    for doc in documents:
        raw_chunks = _split_text(doc.text, c_size, c_overlap)
        source_stem = doc.source.split("/")[-1].replace(".pdf", "")

        for idx, text in enumerate(raw_chunks):
            chunk_id = f"{source_stem}_p{doc.page_number}_c{idx}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=doc.source,
                    page_number=doc.page_number,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                    },
                )
            )

    logger.info(
        "Chunked %d pages → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        c_size,
        c_overlap,
    )
    return all_chunks

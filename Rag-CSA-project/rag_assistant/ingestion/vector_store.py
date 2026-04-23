"""
Vector Store Module
-------------------
Manages a persistent ChromaDB collection.

Responsibilities:
  - Add chunks with their embeddings
  - Query by embedding vector and return top-K results with similarity scores
  - Expose a thin interface so the rest of the system never imports chromadb directly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from rag_assistant.config import config
from rag_assistant.ingestion.chunker import Chunk
from rag_assistant.ingestion.embedder import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search, enriched with its score."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    similarity: float          # cosine similarity in [0, 1]
    metadata: dict


class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Args:
        embedding_model: The embedding backend to use for queries.
        persist_dir: Directory where ChromaDB persists data.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_dir: str = config.chroma_persist_dir,
        collection_name: str = config.chroma_collection_name,
    ) -> None:
        self._embedder = embedding_model
        self._persist_dir = persist_dir
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "ChromaDB collection '%s' ready at %s (docs: %d)",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """
        Embed and store a list of Chunks in ChromaDB.

        Existing documents with the same chunk_id are skipped (idempotent).

        Args:
            chunks: Chunks produced by the chunker.
            batch_size: Number of chunks to embed and upsert per batch.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to do.")
            return

        # Filter out already-stored chunks
        existing_ids = set(self._collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All %d chunks already in store — skipping.", len(chunks))
            return

        logger.info("Embedding %d new chunks…", len(new_chunks))

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed_texts(texts)

            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.metadata for c in batch],
            )
            logger.debug("Upserted batch %d–%d", i, i + len(batch) - 1)

        logger.info("Stored %d chunks in ChromaDB.", len(new_chunks))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = config.top_k_results,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-K most similar chunks for a query.

        Args:
            query_text: The user's question.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by descending similarity.
        """
        if not query_text.strip():
            raise ValueError("query_text must not be empty.")

        query_embedding = self._embedder.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            # For normalised vectors: similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)

            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", "unknown"),
                    text=doc,
                    source=meta.get("source", ""),
                    page_number=int(meta.get("page", 0)),
                    similarity=round(similarity, 4),
                    metadata=meta,
                )
            )

        # Sort descending by similarity (should already be sorted, but be explicit)
        retrieved.sort(key=lambda r: r.similarity, reverse=True)
        logger.debug(
            "Query returned %d chunks; top similarity=%.3f",
            len(retrieved),
            retrieved[0].similarity if retrieved else 0.0,
        )
        return retrieved

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """
        Delete and recreate the collection.
        WARNING: This is destructive and irreversible.
        """
        logger.warning("Resetting ChromaDB collection '%s'.", self._collection_name)
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

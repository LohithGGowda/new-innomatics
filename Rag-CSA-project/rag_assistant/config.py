"""
Central configuration for the RAG Customer Support Assistant.
All tuneable parameters live here to avoid magic numbers scattered across modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Embedding / LLM provider selection
# ---------------------------------------------------------------------------
# Set EMBEDDING_PROVIDER="huggingface" to run fully offline (no API key needed).
# Set EMBEDDING_PROVIDER="openai" to use text-embedding-3-small (requires OPENAI_API_KEY).
EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = os.getenv(
    "EMBEDDING_PROVIDER", "huggingface"
)

# OpenAI settings (only used when EMBEDDING_PROVIDER="openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

# HuggingFace settings (offline fallback)
HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = "rag_support_kb"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 512          # characters per chunk
CHUNK_OVERLAP: int = 64        # overlap to preserve cross-chunk context

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RESULTS: int = 5         # number of chunks to retrieve per query
SIMILARITY_THRESHOLD: float = 0.70  # cosine similarity below this → HITL escalation

# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------
HITL_TIMEOUT_SECONDS: int = 300   # seconds to wait for human input before auto-fail

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


@dataclass(frozen=True)
class AppConfig:
    """Immutable snapshot of runtime configuration."""

    embedding_provider: str = EMBEDDING_PROVIDER
    openai_api_key: str = OPENAI_API_KEY
    openai_embedding_model: str = OPENAI_EMBEDDING_MODEL
    openai_chat_model: str = OPENAI_CHAT_MODEL
    hf_embedding_model: str = HF_EMBEDDING_MODEL
    chroma_persist_dir: str = CHROMA_PERSIST_DIR
    chroma_collection_name: str = CHROMA_COLLECTION_NAME
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k_results: int = TOP_K_RESULTS
    similarity_threshold: float = SIMILARITY_THRESHOLD
    hitl_timeout_seconds: int = HITL_TIMEOUT_SECONDS
    log_level: str = LOG_LEVEL


# Singleton used across the application
config = AppConfig()

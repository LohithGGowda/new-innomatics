"""
Embedding Module
----------------
Provides a unified EmbeddingModel interface that wraps either:
  - OpenAI text-embedding-3-small  (requires OPENAI_API_KEY)
  - HuggingFace sentence-transformers/all-MiniLM-L6-v2  (fully offline)

The active backend is selected via config.embedding_provider.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from rag_assistant.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingModel(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors, one per input text."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Return a single embedding vector for a query string."""


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIEmbeddingModel(EmbeddingModel):
    """Wraps OpenAI's embedding API."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            ) from exc

        if not config.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_embedding_model
        logger.info("OpenAI embedding model: %s", self._model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Wraps sentence-transformers for fully offline embedding."""

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers package is required for HuggingFace embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_name = config.hf_embedding_model
        logger.info("Loading HuggingFace model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_model() -> EmbeddingModel:
    """
    Return the configured embedding model instance.

    Selection is driven by config.embedding_provider:
      - "openai"      → OpenAIEmbeddingModel
      - "huggingface" → HuggingFaceEmbeddingModel (default)
    """
    provider = config.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbeddingModel()
    elif provider == "huggingface":
        return HuggingFaceEmbeddingModel()
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            "Choose 'openai' or 'huggingface'."
        )

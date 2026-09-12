"""Dependency providers. Tests override these via app.dependency_overrides.

Everything here is cached and constructed lazily. The embedder and reranker hold no model
until first use, so importing this module loads no weights (PRD NFR-4).
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import get_settings
from rag.generation.answerer import Answerer
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder
from rag.models.llm import OllamaClient
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore(get_settings())


@lru_cache(maxsize=1)
def get_llm() -> OllamaClient:
    return OllamaClient(get_settings())


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        settings, get_store(), Embedder(settings), CrossEncoderReranker(settings)
    )


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    return Answerer(get_settings(), get_retriever(), get_llm())

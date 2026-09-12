"""Dependency providers. Tests override these via app.dependency_overrides.

Everything here is cached and constructed lazily. The embedder and reranker hold no model
until first use, so importing this module loads no weights (PRD NFR-4).
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import get_settings
from rag.generation.answerer import Answerer
from rag.guardrails.factory import build_pipeline, cached_centroid_provider
from rag.guardrails.guarded import GuardedAnswerer
from rag.guardrails.pipeline import GuardrailPipeline
from rag.guardrails.policy import GuardrailPolicy, load_policy
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


@lru_cache(maxsize=1)
def get_policy() -> GuardrailPolicy:
    return load_policy()


@lru_cache(maxsize=1)
def get_pipeline() -> GuardrailPipeline:
    settings = get_settings()
    return build_pipeline(
        settings,
        get_policy(),
        embedder=Embedder(settings),
        centroid_provider=cached_centroid_provider(),
        # The T3 self-check reuses the generation model. A separate judge would be a
        # second multi-gigabyte resident model for a path that fires on a minority of
        # requests (NFR-3 caps it at 10%).
        judge=get_llm(),
    )


@lru_cache(maxsize=1)
def get_guarded_answerer() -> GuardedAnswerer:
    return GuardedAnswerer(get_answerer(), get_pipeline())

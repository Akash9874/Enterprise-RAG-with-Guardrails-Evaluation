"""Dependency providers. Tests override these via app.dependency_overrides."""

from __future__ import annotations

from functools import lru_cache

from rag.config import Settings, get_settings
from rag.index.qdrant_store import QdrantStore
from rag.models.llm import OllamaClient


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore(get_settings())


@lru_cache(maxsize=1)
def get_llm() -> OllamaClient:
    return OllamaClient(get_settings())


def get_config() -> Settings:
    return get_settings()

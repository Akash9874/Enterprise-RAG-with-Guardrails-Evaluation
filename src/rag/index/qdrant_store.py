"""Qdrant connection and collection lifecycle. Search lands in Phase 1."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient

from rag.config import Settings

log = structlog.get_logger(__name__)


class QdrantStore:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client if client is not None else QdrantClient(url=settings.qdrant.url)

    @property
    def client(self) -> Any:
        return self._client

    def is_ready(self) -> bool:
        try:
            self._client.get_collections()
        except Exception as exc:  # noqa: BLE001 - readiness must never raise
            log.warning("qdrant_not_ready", error=str(exc))
            return False
        return True

    def collection_exists(self) -> bool:
        try:
            collections = self._client.get_collections().collections
        except Exception:  # noqa: BLE001 - readiness must never raise
            return False
        return any(c.name == self._settings.qdrant.collection for c in collections)

"""Qdrant connection and collection lifecycle. Search lands in Phase 1."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient, models

from rag.config import Settings
from rag.contracts import Chunk
from rag.index.schema import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    chunk_to_payload,
    point_id,
    sparse_vectors_config,
    vectors_config,
)

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

    def ensure_collection(self, recreate: bool = False) -> None:
        name = self._settings.qdrant.collection
        if recreate and self.collection_exists():
            self._client.delete_collection(name)
        if not self.collection_exists():
            self._client.create_collection(
                collection_name=name,
                vectors_config=vectors_config(self._settings.qdrant.vector_size),
                sparse_vectors_config=sparse_vectors_config(),
            )
            log.info("collection_created", collection=name)

    def upsert_chunks(self, chunks: list[Chunk], embedder: Any, batch_size: int = 64) -> int:
        if not chunks:
            return 0
        self.ensure_collection()
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            dense = embedder.embed_documents(texts)
            sparse = embedder.embed_sparse(texts)

            points = [
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR: dense_vec,
                        SPARSE_VECTOR: models.SparseVector(indices=indices, values=values),
                    },
                    payload=chunk_to_payload(chunk),
                )
                for chunk, dense_vec, (indices, values) in zip(batch, dense, sparse, strict=True)
            ]
            self._client.upsert(collection_name=self._settings.qdrant.collection, points=points)
            total += len(points)
            log.info("upserted_batch", count=len(points), total=total)
        return total

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        result = self._client.count(collection_name=self._settings.qdrant.collection, exact=True)
        return int(result.count)

"""Qdrant connection and collection lifecycle. Search lands in Phase 1."""

from __future__ import annotations

from typing import Any, cast

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

    def dense_centroid(self, batch_size: int = 256) -> list[float] | None:
        """Mean of every indexed dense vector — the corpus's centre of mass.

        The topicality rail compares a query against this to decide whether it is in
        scope at all (ADR-015). Pages through the whole collection deliberately: a
        centroid computed from the first batch would be a centroid of whatever Qdrant
        happened to return first, which is not the same thing and would not look wrong.
        """
        if not self.collection_exists():
            return None

        total: list[float] | None = None
        count = 0
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._settings.qdrant.collection,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=[DENSE_VECTOR],
            )
            for point in points:
                vectors = point.vector or {}
                raw = vectors[DENSE_VECTOR] if isinstance(vectors, dict) else vectors
                # The client's union type covers sparse and multi-vector shapes too; a
                # collection created by `vectors_config` only ever holds flat floats here.
                vector = [float(v) for v in cast(list[Any], raw)]
                if total is None:
                    total = [0.0] * len(vector)
                for i, value in enumerate(vector):
                    total[i] += value
                count += 1
            if offset is None:
                break

        if total is None or count == 0:
            return None
        return [value / count for value in total]

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int,
        exclude_quarantined: bool = True,
    ) -> list[Any]:
        query_filter = None
        if exclude_quarantined:
            query_filter = models.Filter(
                must_not=[
                    models.FieldCondition(key="quarantined", match=models.MatchValue(value=True))
                ]
            )

        response = self._client.query_points(
            collection_name=self._settings.qdrant.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    limit=self._settings.retrieval.k_dense,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR,
                    limit=self._settings.retrieval.k_sparse,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return list(response.points)

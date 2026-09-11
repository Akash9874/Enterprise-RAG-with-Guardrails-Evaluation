"""Hybrid retrieval: dense + sparse, fused server-side by RRF, then reranked."""

from __future__ import annotations

import time
from typing import Any

import structlog

from rag.config import Settings
from rag.contracts import Retrieved
from rag.index.schema import payload_to_chunk

log = structlog.get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: Any,
        embedder: Any,
        reranker: Any | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self.last_timings: dict[str, float] = {}

    def search(
        self,
        query: str,
        k: int | None = None,
        rerank: bool | None = None,
    ) -> list[Retrieved]:
        cfg = self._settings.retrieval
        k_final = k if k is not None else cfg.k_final
        do_rerank = cfg.rerank_enabled if rerank is None else rerank
        timings: dict[str, float] = {}

        started = time.perf_counter()
        dense = self._embedder.embed_query(query)
        sparse_indices, sparse_values = self._embedder.embed_sparse([query])[0]
        timings["embed_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        points = self._store.hybrid_search(
            dense_vector=dense,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            limit=cfg.k_fuse,
        )
        timings["fuse_ms"] = (time.perf_counter() - started) * 1000

        results = [
            Retrieved(chunk=payload_to_chunk(p.payload), fused_score=float(p.score)) for p in points
        ]
        # Deterministic ordering: fused score desc, then chunk_id asc to break ties.
        results.sort(key=lambda r: (-r.fused_score, r.chunk.chunk_id))

        if do_rerank and self._reranker is not None and results:
            started = time.perf_counter()
            scored = self._reranker.rerank(query, results)
            timings["rerank_ms"] = (time.perf_counter() - started) * 1000
            for item, score in scored:
                item.rerank_score = float(score)
            results = [item for item, _ in scored]
            results.sort(key=lambda r: (-(r.rerank_score or 0.0), r.chunk.chunk_id))

        results = results[:k_final]
        for position, item in enumerate(results, start=1):
            item.rank = position

        self.last_timings = timings
        log.debug("retrieval_complete", results=len(results), **timings)
        return results

"""Ingest-time chunk enrichment: the injection scan and quarantine (FR-I7).

This is the first of three layers against indirect prompt injection — the corpus is
source code and documentation, so a file can carry adversarial instructions. Catching it
here costs one batched pass at ingest instead of a classifier call on every retrieval.

Takes a scoring function rather than importing `guardrails`, which keeps the module
dependencies in PRD §6.1 intact: `ingest` depends on `models` and `contracts`, not on
the guardrail pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from rag.contracts import Chunk

log = structlog.get_logger(__name__)

Scorer = Callable[[list[str]], list[float]]

SCAN_FAILED = -1


def quarantine_chunks(chunks: list[Chunk], scorer: Scorer, threshold: float) -> int:
    """Score every chunk, flag those at or above `threshold`, return how many were flagged.

    Returns SCAN_FAILED (-1) if the scorer malfunctioned. Failing open is deliberate: the
    classifier is one of three layers, and an ingest that dies on a model error is worse
    than one that indexes with the scan skipped and says so.
    """
    if not chunks:
        return 0

    try:
        # One batched call: 567 chunks at 120 ms each would be 68 s scored individually.
        scores = scorer([chunk.text for chunk in chunks])
    except Exception as exc:  # noqa: BLE001 - a model error must not abort ingestion
        log.warning("injection_scan_failed", error=str(exc), chunks=len(chunks))
        return SCAN_FAILED

    quarantined = 0
    for chunk, score in zip(chunks, scores, strict=True):
        chunk.injection_score = float(score)
        if score >= threshold:
            chunk.quarantined = True
            quarantined += 1
            log.info(
                "chunk_quarantined",
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                score=round(float(score), 4),
            )
    return quarantined

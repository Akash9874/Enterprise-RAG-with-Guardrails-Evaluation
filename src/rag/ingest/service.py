"""Ingest orchestration shared by the CLI and `POST /ingest` (FR-A3, FR-I9).

Store, embedder and scorer are injected: ingest must not import the index or guardrail
modules' internals (CLAUDE.md invariant 1).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rag.contracts import IngestSummary
from rag.gitinfo import git_commit
from rag.ingest.enrich import SCAN_FAILED, quarantine_chunks
from rag.ingest.pipeline import build_chunks


def run_ingest(
    source: Path,
    *,
    store: Any,
    embedder: Any,
    recreate: bool = False,
    scorer: Callable[[list[str]], list[float]] | None = None,
    threshold: float = 0.8,
    source_label: str | None = None,
) -> IngestSummary:
    started = time.perf_counter()
    chunks, stats = build_chunks(source)
    # Stamp the commit of the tree being indexed onto every chunk, so provenance can be read
    # back from the index itself (ADR-026).
    commit = git_commit(source)
    for chunk in chunks:
        chunk.corpus_commit = commit

    scan: Literal["ok", "skipped", "failed"] = "skipped"
    quarantined: int | None = None
    if scorer is not None:
        flagged = quarantine_chunks(chunks, scorer=scorer, threshold=threshold)
        if flagged == SCAN_FAILED:
            scan = "failed"
        else:
            scan, quarantined = "ok", flagged

    store.ensure_collection(recreate=recreate)
    upserted = store.upsert_chunks(chunks, embedder)
    return IngestSummary(
        source=source_label or str(source),
        files=stats.files,
        chunks=stats.chunks,
        upserted=int(upserted),
        skipped=stats.skipped,
        quarantined=quarantined,
        scan=scan,
        # Whole-run wall time including the scan and embedding, not just chunking.
        duration_s=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
        corpus_commit=commit,
    )


def save_summary(summary: IngestSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def load_summary(path: Path) -> IngestSummary | None:
    if not path.exists():
        return None
    return IngestSummary.model_validate_json(path.read_text(encoding="utf-8"))

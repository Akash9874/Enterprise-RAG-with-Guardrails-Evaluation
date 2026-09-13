"""POST /ingest and GET /corpus/stats (FR-A3, FR-A4)."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from rag.api.deps import InjectionScorer, get_embedder, get_injection_scorer, get_store
from rag.config import Settings, get_settings, project_path
from rag.contracts import CorpusStats, IngestSummary
from rag.index.qdrant_store import QdrantStore
from rag.ingest.service import load_summary, run_ingest, save_summary
from rag.models.embedder import Embedder

router = APIRouter()
_ingest_lock = threading.Lock()


class IngestRequest(BaseModel):
    source: str = Field(
        default="self",
        description="A key in settings.ingest.sources — never a filesystem path.",
    )
    recreate: bool = False
    scan: bool = True


@router.post("/ingest", response_model=IngestSummary)
def ingest(
    request: IngestRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[QdrantStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    scorer: Annotated[InjectionScorer, Depends(get_injection_scorer)],
) -> IngestSummary:
    """Re-index a configured source. Synchronous: this corpus ingests in seconds to minutes,
    and a job queue would be scope without signal (PRD §3.2)."""
    root = settings.ingest.sources.get(request.source)
    if root is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown source {request.source!r}; configured: "
            f"{sorted(settings.ingest.sources)}",
        )
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an ingest is already running")
    try:
        summary = run_ingest(
            project_path(root),
            store=store,
            embedder=embedder,
            recreate=request.recreate,
            scorer=scorer.score_texts if request.scan else None,
            threshold=scorer.threshold,
            source_label=request.source,
        )
        save_summary(summary, project_path(settings.ingest.state_path))
    finally:
        _ingest_lock.release()
    return summary


@router.get("/corpus/stats", response_model=CorpusStats)
def corpus_stats(
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[QdrantStore, Depends(get_store)],
) -> CorpusStats:
    exists = store.collection_exists()
    languages: Counter[str] = Counter()
    quarantined = 0
    if exists:
        # A payload scroll, not a facet: facets need a keyword payload index this collection
        # does not have, and ~1,000 points page through in milliseconds.
        for payload in store.iter_payloads(["language", "quarantined"]):
            languages[str(payload.get("language", "unknown"))] += 1
            quarantined += bool(payload.get("quarantined"))
    return CorpusStats(
        collection=settings.qdrant.collection,
        exists=exists,
        points=sum(languages.values()),
        by_language=dict(sorted(languages.items())),
        quarantined=quarantined,
        last_ingest=load_summary(project_path(settings.ingest.state_path)),
    )

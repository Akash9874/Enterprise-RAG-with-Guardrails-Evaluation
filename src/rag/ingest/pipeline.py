"""Orchestrates load -> chunk -> dedupe."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from rag.contracts import Chunk
from rag.ingest.chunkers.code import chunk_code
from rag.ingest.chunkers.markdown import chunk_markdown
from rag.ingest.loaders import load_repository

log = structlog.get_logger(__name__)

MARKDOWN_LANGUAGES = {"markdown"}


@dataclass(frozen=True)
class IngestStats:
    files: int
    chunks: int
    skipped: int
    duration_s: float


def build_chunks(root: Path) -> tuple[list[Chunk], IngestStats]:
    started = time.perf_counter()
    seen: set[str] = set()
    chunks: list[Chunk] = []
    files = 0
    skipped = 0

    for file in load_repository(root):
        files += 1
        produced = chunk_markdown(file) if file.language in MARKDOWN_LANGUAGES else chunk_code(file)
        if not produced:
            skipped += 1
            continue
        for chunk in produced:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)

    stats = IngestStats(
        files=files,
        chunks=len(chunks),
        skipped=skipped,
        duration_s=time.perf_counter() - started,
    )
    log.info("ingest_complete", **stats.__dict__)
    return chunks, stats

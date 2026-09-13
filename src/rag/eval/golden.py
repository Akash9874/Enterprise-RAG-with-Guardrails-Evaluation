"""Golden query sets.

`provenance` is mandatory. Hand-authored and synthetic questions are scored separately:
synthetic questions are generated FROM the chunk that answers them, so their lexical
overlap is artificially high and pooling the two would flatter the system (PRD FR-E1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Provenance = Literal["hand", "synthetic"]


class GoldenQuery(BaseModel):
    id: str
    query: str
    provenance: Provenance
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    golden_answer: str | None = None
    expect_refusal: bool = False
    # Synthetic only: set true once a human has verified the entry (PRD §7.5, 20%).
    spot_checked: bool | None = None

    @model_validator(mode="after")
    def require_ground_truth(self) -> GoldenQuery:
        if not self.expect_refusal and not self.relevant_chunk_ids and not self.relevant_files:
            raise ValueError(
                f"{self.id}: needs ground truth — set relevant_chunk_ids, "
                "relevant_files, or expect_refusal"
            )
        return self


def load_golden(path: Path) -> list[GoldenQuery]:
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    queries = [GoldenQuery.model_validate(entry) for entry in raw]

    seen: set[str] = set()
    for query in queries:
        if query.id in seen:
            raise ValueError(f"duplicate golden query id: {query.id}")
        seen.add(query.id)
    return queries


def golden_set_hash(path: Path) -> str:
    """Content hash of a golden file, normalised to LF so every checkout agrees."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.blake2b(text.encode(), digest_size=6).hexdigest()


def stale_chunk_refs(queries: list[GoldenQuery], indexed_ids: set[str]) -> dict[str, list[str]]:
    """Golden chunk ids absent from the index.

    `chunk_id` is a content hash, so editing a referenced chunk orphans the reference.
    Scoring an orphan as a miss would report a retrieval regression that never happened,
    so callers fail loudly instead.
    """
    stale: dict[str, list[str]] = {}
    for query in queries:
        missing = sorted(set(query.relevant_chunk_ids) - indexed_ids)
        if missing:
            stale[query.id] = missing
    return stale

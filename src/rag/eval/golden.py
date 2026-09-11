"""Golden query sets.

`provenance` is mandatory. Hand-authored and synthetic questions are scored separately:
synthetic questions are generated FROM the chunk that answers them, so their lexical
overlap is artificially high and pooling the two would flatter the system (PRD FR-E1).
"""

from __future__ import annotations

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

"""Types crossing module boundaries. No module may import another module's internals."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

Tier = Literal["T0", "T1", "T2", "T3"]
Verdict = Literal["pass", "hedge", "redact", "refuse", "block", "skipped", "error"]


def make_chunk_id(source_path: str, text: str) -> str:
    """Stable, content-derived id.

    Re-ingesting unchanged content yields the same id, which is what keeps ingestion
    idempotent and golden-set references valid across re-indexing.
    """
    digest = hashlib.blake2b(f"{source_path}\x00{text}".encode(), digest_size=16)
    return digest.hexdigest()


class PIIFinding(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    language: str
    symbol_path: str | None = None
    header_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    token_count: int = 0
    pii_findings: list[PIIFinding] = Field(default_factory=list)
    quarantined: bool = False
    # Injection probability recorded at ingest (FR-I7). Kept even when below the
    # quarantine threshold, so the decision is reviewable rather than just a boolean.
    injection_score: float | None = None
    content_hash: str = ""


class Retrieved(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    rank: int = 0


class Citation(BaseModel):
    marker: str
    chunk_id: str
    source_path: str
    display_path: str
    supported: bool | None = None


class RailResult(BaseModel):
    rail: str
    tier: Tier
    verdict: Verdict
    score: float | None = None
    threshold_band: tuple[float, float] | None = None
    latency_ms: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)


class RailContext(BaseModel):
    """What a rail is allowed to see. Rails never mutate it; they return a RailResult."""

    request_id: str
    query: str
    answer: str | None = None
    retrieved: list[Retrieved] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class GuardrailTrace(BaseModel):
    request_id: str
    input_rails: list[RailResult] = Field(default_factory=list)
    output_rails: list[RailResult] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: str | None = None
    final_verdict: str = "pass"
    total_latency_ms: float = 0.0
    budget_exceeded: bool = False


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[Retrieved] = Field(default_factory=list)
    # Set by generation, and distinct from the guardrail trace below: `refused` is the
    # empty-retrieval refusal of FR-G6, `ungrounded` and `stripped_markers` are what
    # citation enforcement (FR-G4) found. The output rails consume all three.
    refused: bool = False
    ungrounded: bool = False
    stripped_markers: list[str] = Field(default_factory=list)
    trace: GuardrailTrace | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    model_info: dict[str, str] = Field(default_factory=dict)

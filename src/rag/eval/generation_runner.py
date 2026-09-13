"""Tier B runner. Answers come from the real guarded pipeline; scoring uses no LLM (FR-E2).

Wall time is dominated by generation — ~35 s p50 per query on the reference hardware — so a
full golden set takes tens of minutes. That is why CI does not run it (ADR-026).
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from rag.eval.golden import GoldenQuery
from rag.eval.metrics.generation import (
    RefusalOutcome,
    bertscore_f1,
    citation_precision,
    citation_recall,
    groundedness,
    refusal_correctness,
    refusal_outcome,
)
from rag.eval.scorers import split_answer
from rag.guardrails.guarded import HEDGE_PREFIX
from rag.guardrails.rails.groundedness import strip_markers

PROVENANCES = ("hand", "synthetic")
OUTCOMES: tuple[RefusalOutcome, ...] = (
    "correct_answer",
    "correct_refusal",
    "false_refusal",
    "missed_refusal",
)


class TierBResult(BaseModel):
    by_provenance: dict[str, dict[str, float]]
    counts: dict[str, dict[str, int]]
    per_query: list[dict[str, Any]] = Field(default_factory=list)


def run_tier_b(
    queries: list[GoldenQuery],
    answerer: Any,
    support_scorer: Any,
    token_embedder: Any,
    t_pass: float,
    progress: bool = False,
) -> TierBResult:
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        answer = answerer.answer(query.query)
        row: dict[str, Any] = {
            "id": query.id,
            "query": query.query,
            "provenance": query.provenance,
            "expect_refusal": query.expect_refusal,
            "refused": answer.refused,
            "refusal_outcome": refusal_outcome(query.expect_refusal, answer.refused),
            "answer": answer.text,
            "contexts": [r.chunk.text for r in answer.retrieved],
            "citations": [c.marker for c in answer.citations],
            "ungrounded": answer.ungrounded,
            "stripped_markers": answer.stripped_markers,
            "final_verdict": answer.trace.final_verdict if answer.trace else None,
        }
        # Generation metrics only for questions that should be answered and were.
        if not answer.refused and not query.expect_refusal:
            text = answer.text.removeprefix(HEDGE_PREFIX)
            by_id = {r.chunk.chunk_id: r.chunk.text for r in answer.retrieved}
            chunk_by_marker = {
                c.marker: by_id[c.chunk_id] for c in answer.citations if c.chunk_id in by_id
            }
            scored = support_scorer.score(split_answer(text), chunk_by_marker)
            row["sentences"] = [s.model_dump() for s in scored]
            row["groundedness"] = groundedness(scored)
            row["citation_precision"] = citation_precision(scored, t_pass)
            row["citation_recall"] = citation_recall(scored, t_pass)
            if query.golden_answer:
                candidate, reference = token_embedder.embed(
                    [strip_markers(text), query.golden_answer]
                )
                row["bertscore_f1"] = bertscore_f1(candidate, reference)
        rows.append(row)
        if progress:
            print(f"  [{index}/{len(queries)}] {query.id}: {row['refusal_outcome']}", flush=True)
    return _aggregate(rows)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> TierBResult:
    by_provenance: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for provenance in PROVENANCES:
        mine = [r for r in rows if r["provenance"] == provenance]
        scored = [r for r in mine if "groundedness" in r]
        precision = [r["citation_precision"] for r in scored if r["citation_precision"] is not None]
        bert = [r["bertscore_f1"] for r in scored if "bertscore_f1" in r]
        metrics: dict[str, float | None] = {
            "groundedness": _mean([r["groundedness"] for r in scored]),
            "citation_precision": _mean(precision),
            "citation_recall": _mean([r["citation_recall"] for r in scored]),
            "bertscore_f1": _mean(bert),
            "refusal_correctness": (
                refusal_correctness([r["refusal_outcome"] for r in mine]) if mine else None
            ),
            "ungrounded_rate": _mean([1.0 if r["ungrounded"] else 0.0 for r in scored]),
        }
        # A metric with no data is omitted rather than zeroed; the gate reads it as not run.
        by_provenance[provenance] = {k: v for k, v in metrics.items() if v is not None}
        counts[provenance] = {
            "queries": len(mine),
            "scored": len(scored),
            "with_citations": len(precision),
            "with_golden_answer": len(bert),
            **{o: sum(1 for r in mine if r["refusal_outcome"] == o) for o in OUTCOMES},
        }
    return TierBResult(by_provenance=by_provenance, counts=counts, per_query=rows)

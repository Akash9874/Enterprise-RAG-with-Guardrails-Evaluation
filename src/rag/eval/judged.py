"""Tier C — LLM-judged, opt-in (FR-E3). Supplementary, never the foundation.

Scores are only meaningful alongside the judge that produced them, so the judge's name is
part of the result and of the report provenance. A judge failure is counted, never zeroed:
a 3B model often breaks structured-output parsing, and hiding that would flatter the tier.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from rag.guardrails.guarded import HEDGE_PREFIX

HALVES = ("hand", "synthetic")
METRICS = ("faithfulness", "answer_relevancy")


class Judge(Protocol):
    name: str

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float: ...

    def answer_relevancy(self, question: str, answer: str) -> float: ...


class TierCResult(BaseModel):
    judge: str
    by_provenance: dict[str, dict[str, float]]
    counts: dict[str, dict[str, int]]
    per_query: list[dict[str, Any]] = Field(default_factory=list)


def _judge_row(row: dict[str, Any], judge: Judge) -> dict[str, Any]:
    question = str(row["query"])
    answer = str(row["answer"]).removeprefix(HEDGE_PREFIX)
    contexts = list(row["contexts"])
    calls: dict[str, Callable[[], float]] = {
        "faithfulness": lambda: judge.faithfulness(question, answer, contexts),
        "answer_relevancy": lambda: judge.answer_relevancy(question, answer),
    }
    out: dict[str, Any] = {"id": row["id"], "provenance": row["provenance"]}
    for metric, call in calls.items():
        try:
            out[metric] = float(call())
        except Exception as exc:  # noqa: BLE001 - a failed judgement is counted, not hidden
            out[f"{metric}_error"] = type(exc).__name__
    return out


def run_tier_c(rows: list[dict[str, Any]], judge: Judge, limit: int | None = None) -> TierCResult:
    """Judge Tier B's stored answers. Only rows Tier B scored (answered, should-answer)."""
    eligible = [r for r in rows if "groundedness" in r]
    if limit is not None:
        eligible = eligible[:limit]
    per_query = [_judge_row(row, judge) for row in eligible]

    by_provenance: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for half in HALVES:
        mine = [q for q in per_query if q["provenance"] == half]
        by_provenance[half] = {
            metric: statistics.fmean(q[metric] for q in mine if metric in q)
            for metric in METRICS
            if any(metric in q for q in mine)
        }
        counts[half] = {"judged": len(mine)} | {
            f"{metric}_failed": sum(1 for q in mine if f"{metric}_error" in q) for metric in METRICS
        }
    return TierCResult(
        judge=judge.name, by_provenance=by_provenance, counts=counts, per_query=per_query
    )

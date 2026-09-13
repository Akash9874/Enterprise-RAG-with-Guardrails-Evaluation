"""Shared builders for report and gate tests. Not a test module."""

from __future__ import annotations

from rag.eval.provenance import Provenance
from rag.eval.report import EvalReport
from rag.eval.runner import TierAResult

GATE = {
    "tier_a.by_provenance.hand.recall@5": 0.02,
    "tier_b.by_provenance.hand.groundedness": 0.03,
}


def make_provenance(commit: str = "abc1234", golden: str = "g1") -> Provenance:
    return Provenance(
        corpus_commit=commit,
        config_hash="c",
        policy_hash="p",
        golden_set_hash=golden,
        golden_set={"hand": 1, "synthetic": 0},
        models={"embedder": "e"},
    )


def make_report(
    hand_recall: float | None, golden: str = "g1", commit: str = "abc1234"
) -> EvalReport:
    tier_a = None
    if hand_recall is not None:
        tier_a = TierAResult(
            overall={},
            by_provenance={"hand": {"recall@5": hand_recall}, "synthetic": {}},
            reranker_lift=None,
            scored_queries=1,
        )
    return EvalReport(provenance=make_provenance(commit=commit, golden=golden), tier_a=tier_a)

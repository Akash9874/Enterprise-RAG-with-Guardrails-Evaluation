"""Tier B generation metrics. Pure functions — no model loading, no I/O (PRD FR-E2).

Scores arrive precomputed in `ScoredSentence.support`; the model adapters that produce
them live in `rag.eval.scorers`, so everything here is testable with hand-built inputs.
"""

from __future__ import annotations

import statistics
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

RefusalOutcome = Literal["correct_answer", "correct_refusal", "false_refusal", "missed_refusal"]


class ScoredSentence(BaseModel):
    text: str
    cited_markers: list[str] = Field(default_factory=list)
    # HHEM score of this sentence against each cited chunk, keyed by marker.
    support: dict[str, float] = Field(default_factory=dict)


def groundedness(sentences: list[ScoredSentence]) -> float:
    """Mean over sentences of the best cited-chunk score. Uncited sentences score 0."""
    if not sentences:
        return 0.0
    return statistics.fmean(max(s.support.values(), default=0.0) for s in sentences)


def citation_precision(sentences: list[ScoredSentence], t_pass: float) -> float | None:
    """Fraction of emitted (sentence, cited chunk) pairs at or above t_pass.

    None when nothing is cited: 0/0 is undefined, and scoring it 0 or 1 would both lie.
    """
    scores = [score for s in sentences for score in s.support.values()]
    if not scores:
        return None
    return sum(1 for score in scores if score >= t_pass) / len(scores)


def citation_recall(sentences: list[ScoredSentence], t_pass: float) -> float:
    """Fraction of sentences carrying at least one supporting citation.

    Every sentence counts as a claim — a stated simplification (ADR-024).
    """
    if not sentences:
        return 0.0
    supported = sum(1 for s in sentences if any(v >= t_pass for v in s.support.values()))
    return supported / len(sentences)


def refusal_outcome(expect_refusal: bool, refused: bool) -> RefusalOutcome:
    if expect_refusal:
        return "correct_refusal" if refused else "missed_refusal"
    return "false_refusal" if refused else "correct_answer"


def refusal_correctness(outcomes: list[RefusalOutcome]) -> float:
    if not outcomes:
        return 0.0
    correct = sum(1 for o in outcomes if o in {"correct_answer", "correct_refusal"})
    return correct / len(outcomes)


def bertscore_f1(candidate: npt.ArrayLike, reference: npt.ArrayLike) -> float:
    """BERTScore F1 by greedy cosine matching (Zhang et al., 2020).

    No IDF weighting and no baseline rescaling — the `bert-score` package defaults, so the
    parity check against that package compares like with like. Special tokens are removed
    by the caller.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.ndim != 2 or ref.ndim != 2 or not len(cand) or not len(ref):
        return 0.0
    cand = cand / np.clip(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12, None)
    ref = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
    sim = cand @ ref.T
    precision = float(sim.max(axis=1).mean())
    recall = float(sim.max(axis=0).mean())
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

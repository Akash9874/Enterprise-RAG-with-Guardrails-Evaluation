"""T2 groundedness rail with T3 escalation (FR-GR1, FR-GR4, FR-GR7).

HHEM-2.1-Open scores each answer sentence against the chunks that sentence cites. One
model serves as both this rail and the Tier B eval metric (ADR-005), so the thing being
measured is definitionally the thing being enforced.

**This is the only rail permitted to reach T3.** Escalation fires only from inside the
band, and only while the per-request budget lasts. Input rails resolve on score alone.
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from rag.contracts import RailContext, RailResult, Tier, Verdict
from rag.guardrails.base import timed
from rag.guardrails.policy import RailPolicy

log = structlog.get_logger(__name__)

MODEL_NAME = "vectara/hallucination_evaluation_model"
_MARKER = re.compile(r"\[(\d+)\]")
_MARKER_GROUP = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# A terminator followed by whitespace and something that starts a new sentence. The
# lookbehind excludes a digit so that "-0.106 NDCG" is not split at the decimal point.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\d\.)\s+(?=[A-Z(\[])")

JUDGE_SYSTEM = (
    "You judge whether a claim is supported by a source passage. Reply with exactly one "
    "word: SUPPORTED or UNSUPPORTED. Do not explain."
)


def strip_markers(text: str) -> str:
    """Remove citation markers before scoring.

    MEASURED 2026-09-12: leaving them in costs ~0.10-0.12 on every supported sentence
    (0.944 -> 0.846, 0.970 -> 0.849, 0.965 -> 0.869) while a contradicted sentence is
    unaffected (0.507 -> 0.511). The marker is not part of the claim, and keeping it
    compresses exactly the separation this rail depends on. See ADR-021.
    """
    return _MARKER_GROUP.sub("", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences for per-sentence scoring."""
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENTENCE_END.split(stripped) if part.strip()]


class GroundednessRail:
    name = "groundedness"
    tier: Tier = "T2"

    def __init__(
        self,
        policy: RailPolicy,
        hhem: Any | None = None,
        judge: Any | None = None,
        escalation_budget_ms: int = 5000,
    ) -> None:
        self._policy = policy
        self._hhem = hhem
        self._judge = judge
        self._budget_ms = escalation_budget_ms

    @property
    def hhem(self) -> Any:
        if self._hhem is None:
            from transformers import AutoModelForSequenceClassification

            self._hhem = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            log.info("hhem_loaded", model=MODEL_NAME)
        return self._hhem

    def check(self, ctx: RailContext) -> RailResult:
        with timed() as elapsed:
            evidence: dict[str, Any] = {"model": MODEL_NAME}
            sentences = split_sentences(ctx.answer or "")
            if not sentences:
                return RailResult(
                    rail=self.name,
                    tier="T2",
                    verdict="skipped",
                    latency_ms=elapsed[0],
                    evidence={"reason": "empty_answer"},
                )

            chunks = [r.chunk.text for r in ctx.retrieved if r.chunk.text.strip()]
            scored = self._score_sentences(sentences, chunks)
            mean = sum(score for _, score in scored) / len(scored)

            t_pass = self._policy.t_pass if self._policy.t_pass is not None else 0.65
            t_block = self._policy.t_block if self._policy.t_block is not None else 0.35

            evidence["sentence_scores"] = [
                {"sentence": s, "score": round(score, 4)} for s, score in scored
            ]
            evidence["unsupported_sentences"] = [s for s, score in scored if score < t_block]

            verdict: Verdict
            if mean >= t_pass:
                verdict = "pass"
            elif mean <= t_block:
                # The configured action, not a hardcoded refusal. Groundedness is a
                # quality rail: it degrades the answer rather than denying service
                # (FR-GR5), and hardcoding `refuse` here refused 11 of 12 benign
                # controls in the first full-path run (ADR-022).
                verdict = self._policy.trip_verdict
            else:
                verdict = self._escalate(ctx, scored, mean, evidence)

        return RailResult(
            rail=self.name,
            tier="T2",
            verdict=verdict,
            score=mean,
            threshold_band=self._policy.band,
            latency_ms=elapsed[0],
            evidence=evidence,
        )

    def _score_sentences(self, sentences: list[str], chunks: list[str]) -> list[tuple[str, float]]:
        """Score each sentence against each chunk separately, keeping the best.

        Two measured reasons for this shape (ADR-021):

        1. **Per chunk, never concatenated.** HHEM's window is 512 tokens. Joining the
           retrieved chunks into one premise overflows it and the tail is silently
           truncated - a correct, cited answer scored 0.184 that way against 0.969 when
           scored chunk by chunk.
        2. **Max, not the cited chunk.** This rail asks whether the answer is
           hallucinated, so a claim supported by *any* retrieved passage is grounded.
           Whether it cited the right one is citation precision - a separate metric,
           and conflating the two produced no usable signal at all.
        """
        if not chunks:
            return [(sentence, 0.0) for sentence in sentences]

        hypotheses = [strip_markers(sentence) for sentence in sentences]
        pairs = [(chunk, hypothesis) for hypothesis in hypotheses for chunk in chunks]
        flat = [float(score) for score in self.hhem.predict(pairs)]

        width = len(chunks)
        return [
            (sentence, max(flat[index * width : (index + 1) * width]))
            for index, sentence in enumerate(sentences)
        ]

    def _escalate(
        self,
        ctx: RailContext,
        scored: list[tuple[str, float]],
        mean: float,
        evidence: dict[str, Any],
    ) -> Verdict:
        """T3 LLM self-check — band only, budget-capped (FR-GR4, FR-GR7)."""
        if not self._policy.escalate or self._judge is None:
            evidence["escalated"] = False
            evidence["escalation_reason"] = "disabled" if not self._policy.escalate else "no_judge"
            return "hedge"

        if self._budget_ms <= 0:
            # Never silently skip the budget check: degrade to the T2 verdict and say so.
            evidence["escalated"] = False
            evidence["budget_exceeded"] = True
            evidence["escalation_reason"] = "budget_exhausted"
            return "hedge"

        weakest = min(scored, key=lambda pair: pair[1])[0]
        premise = "\n".join(r.chunk.text for r in ctx.retrieved)
        started = time.perf_counter()
        reply = self._judge.generate(
            f"Source passage:\n{premise}\n\nClaim:\n{weakest}\n\nIs the claim supported?",
            system=JUDGE_SYSTEM,
        )
        spent = (time.perf_counter() - started) * 1000

        evidence["escalated"] = True
        evidence["escalation_reason"] = f"score {mean:.3f} inside band"
        evidence["judge_reply"] = reply.strip()[:80]
        evidence["escalation_ms"] = round(spent, 1)
        if spent > self._budget_ms:
            evidence["budget_exceeded"] = True

        return "pass" if reply.strip().upper().startswith("SUPPORTED") else "hedge"

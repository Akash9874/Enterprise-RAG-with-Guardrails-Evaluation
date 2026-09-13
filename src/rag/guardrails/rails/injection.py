"""T1 prompt-injection rail — `deberta-v3-base-prompt-injection-v2` (FR-GR1).

MEASURED 2026-09-12 on the reference CPU: ~0.89 GB resident, 120 ms warm per query
(306 ms cold). Both are well over the PRD §4.1 estimates of 0.40 GB and ~40 ms — the
budget line was written before the model was run. See ADR-017.

Input rails resolve on score alone (FR-GR4): there is no T3 escalation here. A score
inside the band hedges rather than blocks, because denying on uncertainty is precisely
how a guardrail acquires a false-refusal rate.
"""

from __future__ import annotations

from typing import Any

import structlog

from rag.contracts import RailContext, RailResult, Tier, Verdict
from rag.guardrails.base import timed
from rag.guardrails.policy import RailPolicy

log = structlog.get_logger(__name__)

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
INJECTION_LABEL = "INJECTION"


class InjectionRail:
    name = "injection_input"
    tier: Tier = "T1"

    def __init__(self, policy: RailPolicy, classifier: Any | None = None) -> None:
        self._policy = policy
        self._classifier = classifier

    @property
    def classifier(self) -> Any:
        if self._classifier is None:
            from transformers import pipeline

            # Chunks and queries both exceed 512 tokens sometimes; truncating is correct
            # here because an injection payload that only appears past token 512 of a
            # query is not a realistic attack shape.
            self._classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                device=-1,
                truncation=True,
                max_length=512,
            )
            log.info("injection_classifier_loaded", model=MODEL_NAME)
        return self._classifier

    def score_text(self, text: str) -> tuple[float, str]:
        """Injection probability and the winning label. Shared with the ingest-time scan."""
        (prediction,) = self.classifier(text)
        label = str(prediction["label"]).upper()
        raw = float(prediction["score"])
        # The pipeline reports the confidence of whichever label won, so a SAFE result at
        # 0.99 means an injection probability of 0.01. Reading `raw` directly would invert
        # the rail on every safe input.
        return (raw if label == INJECTION_LABEL else 1.0 - raw), label

    def score_texts(self, texts: list[str]) -> list[float]:
        """Batched injection probabilities. Used by the ingest-time scan (FR-I7)."""
        if not texts:
            return []
        predictions = self.classifier(texts)
        return [
            (
                float(p["score"])
                if str(p["label"]).upper() == INJECTION_LABEL
                else 1.0 - float(p["score"])
            )
            for p in predictions
        ]

    def check(self, ctx: RailContext) -> RailResult:
        with timed() as elapsed:
            probability, label = self.score_text(ctx.query)

        t_pass = self._policy.t_pass if self._policy.t_pass is not None else 0.30
        t_block = self._policy.t_block if self._policy.t_block is not None else 0.80

        verdict: Verdict
        if probability >= t_block:
            verdict = self._policy.trip_verdict  # declared in policy, not hardcoded (FR-GR3)
        elif probability <= t_pass:
            verdict = "pass"
        else:
            verdict = "hedge"

        return RailResult(
            rail=self.name,
            tier="T1",
            verdict=verdict,
            score=probability,
            threshold_band=self._policy.band,
            latency_ms=elapsed[0],
            evidence={"label": label, "model": MODEL_NAME, "scanned_chars": len(ctx.query)},
        )

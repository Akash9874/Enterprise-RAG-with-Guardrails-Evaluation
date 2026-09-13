"""T1 topicality rail — cosine similarity to the corpus centroid (FR-GR1).

ADR-015 moved out-of-scope refusal here after measuring that the RRF fused score cannot
separate in-corpus from out-of-corpus queries at all. Similarity to the corpus centroid
is a calibrated signal; the fused score is rank arithmetic.

Reuses the retrieval embedder, so this rail costs one embed call and adds no model
residency — which is why it can sit at T1 rather than T2.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import structlog

from rag.contracts import RailContext, RailResult, Tier, Verdict
from rag.guardrails.base import timed
from rag.guardrails.policy import RailPolicy

log = structlog.get_logger(__name__)

Centroid = list[float] | None


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, normalising both sides.

    The centroid is a mean of unit vectors and is therefore not itself a unit vector, so
    normalising here rather than assuming it is what keeps the score on [-1, 1].
    """
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


class TopicalityRail:
    name = "topicality"
    tier: Tier = "T1"

    def __init__(
        self,
        policy: RailPolicy,
        embedder: Any,
        centroid_provider: Callable[[], Centroid],
    ) -> None:
        self._policy = policy
        self._embedder = embedder
        self._centroid_provider = centroid_provider

    def check(self, ctx: RailContext) -> RailResult:
        with timed() as elapsed:
            centroid = self._centroid_provider()
            if not centroid:
                # Nothing to compare against. Skipping is right: refusing every query
                # because the corpus is unavailable would be the worse failure.
                return RailResult(
                    rail=self.name,
                    tier="T1",
                    verdict="skipped",
                    latency_ms=elapsed[0],
                    evidence={"reason": "no_centroid"},
                )
            similarity = cosine(self._embedder.embed_query(ctx.query), centroid)

        t_pass = self._policy.t_pass if self._policy.t_pass is not None else 0.45
        t_block = self._policy.t_block if self._policy.t_block is not None else 0.20

        verdict: Verdict
        if similarity >= t_pass:
            verdict = "pass"
        elif similarity <= t_block:
            verdict = self._policy.trip_verdict  # declared in policy, not hardcoded (FR-GR3)
        else:
            verdict = "hedge"

        return RailResult(
            rail=self.name,
            tier="T1",
            verdict=verdict,
            score=similarity,
            threshold_band=self._policy.band,
            latency_ms=elapsed[0],
            evidence={"similarity": round(similarity, 4), "centroid_dims": len(centroid)},
        )

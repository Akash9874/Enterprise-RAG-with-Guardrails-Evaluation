"""Assemble the pipeline in cost order (FR-GR1).

Rail order in these lists *is* the tiering. A rail placed at the wrong position silently
destroys the latency budget while every test still passes, so the order here is asserted
directly in tests rather than left as a convention.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from rag.config import Settings
from rag.guardrails.base import Rail
from rag.guardrails.pipeline import GuardrailPipeline
from rag.guardrails.policy import GuardrailPolicy
from rag.guardrails.rails.groundedness import GroundednessRail
from rag.guardrails.rails.heuristics import InputHeuristicsRail
from rag.guardrails.rails.injection import InjectionRail
from rag.guardrails.rails.pii import PIIRail
from rag.guardrails.rails.topicality import TopicalityRail


def build_pipeline(
    settings: Settings,
    policy: GuardrailPolicy,
    embedder: object,
    centroid_provider: Callable[[], list[float] | None],
    judge: object | None = None,
) -> GuardrailPipeline:
    """Input rails ascend T0 -> T1; output rails ascend T0 -> T2 (-> T3 on escalation)."""
    input_rails: list[Rail] = [
        # T0 — microseconds. A blatant attack never reaches the 120 ms classifier.
        InputHeuristicsRail(policy.for_rail("input_heuristics")),
        PIIRail("pii_input", policy.for_rail("pii_input"), field="query"),
        # T1 — tens of milliseconds.
        InjectionRail(policy.for_rail("injection_input")),
        TopicalityRail(policy.for_rail("topicality"), embedder, centroid_provider),
    ]
    output_rails: list[Rail] = [
        # T0 leak re-scan before the expensive NLI check.
        PIIRail("pii_output", policy.for_rail("pii_output"), field="answer"),
        # T2, and the only rail that may reach T3.
        GroundednessRail(
            policy.for_rail("groundedness"),
            judge=judge,
            escalation_budget_ms=policy.escalation_budget_ms,
            model_name=settings.models.groundedness,
            revision=settings.models.groundedness_revision,
        ),
    ]
    return GuardrailPipeline(policy, input_rails, output_rails)


@lru_cache(maxsize=1)
def cached_centroid_provider() -> Callable[[], list[float] | None]:
    """Compute the corpus centroid once per process.

    It changes only on re-ingest, and recomputing it per query would add a full
    collection scan to every request.
    """
    holder: dict[str, list[float] | None] = {}

    def provider() -> list[float] | None:
        if "centroid" not in holder:
            from rag.api.deps import get_store

            holder["centroid"] = get_store().dense_centroid()
        return holder["centroid"]

    return provider

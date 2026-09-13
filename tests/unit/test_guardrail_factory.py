"""Rail order is the design (FR-GR1).

A rail at the wrong tier position silently destroys the latency budget while every other
test still passes, so the order is asserted here rather than left as a convention.
"""

from __future__ import annotations

from rag.config import Settings
from rag.guardrails.factory import build_pipeline
from rag.guardrails.policy import load_policy


def _pipeline():
    return build_pipeline(
        Settings(),
        load_policy(),
        embedder=object(),
        centroid_provider=lambda: None,
        judge=None,
    )


def _tiers(rails) -> list[str]:
    return [rail.tier for rail in rails]


def test_input_rails_run_in_ascending_cost_order() -> None:
    rails = _pipeline()._input_rails  # type: ignore[attr-defined]
    assert [r.name for r in rails] == [
        "input_heuristics",
        "pii_input",
        "injection_input",
        "topicality",
    ]


def test_input_tiers_never_decrease() -> None:
    assert _tiers(_pipeline()._input_rails) == sorted(_tiers(_pipeline()._input_rails))  # type: ignore[attr-defined]


def test_the_microsecond_rails_precede_the_classifier() -> None:
    """A blatant attack must never reach the 120 ms classifier (ADR-018)."""
    rails = _pipeline()._input_rails  # type: ignore[attr-defined]
    names = [r.name for r in rails]
    assert names.index("input_heuristics") < names.index("injection_input")


def test_output_rails_put_the_cheap_leak_scan_before_the_nli_check() -> None:
    rails = _pipeline()._output_rails  # type: ignore[attr-defined]
    assert [r.name for r in rails] == ["pii_output", "groundedness"]
    assert _tiers(rails) == ["T0", "T2"]


def test_only_the_groundedness_rail_can_escalate() -> None:
    """FR-GR4. Escalation anywhere else is a PRD change, not a config change."""
    policy = load_policy()
    escalating = {name for name, rail in policy.rails.items() if rail.escalate}
    assert escalating == {"groundedness"}


def test_building_the_pipeline_loads_no_model() -> None:
    """Rails hold loaders, not models. Import-time construction breaks the RAM budget."""
    pipeline = _pipeline()
    for rail in pipeline._input_rails + pipeline._output_rails:  # type: ignore[attr-defined]
        for attribute in ("_analyzer", "_classifier", "_hhem"):
            assert getattr(rail, attribute, None) is None


def test_groundedness_rail_gets_its_model_and_revision_from_settings() -> None:
    """Never a hardcoded model name (CLAUDE.md), and remote code is always pinned."""
    settings = Settings()
    rail = next(r for r in _pipeline()._output_rails if r.name == "groundedness")  # type: ignore[attr-defined]
    assert rail._model_name == settings.models.groundedness  # type: ignore[attr-defined]
    assert rail._revision == settings.models.groundedness_revision  # type: ignore[attr-defined]

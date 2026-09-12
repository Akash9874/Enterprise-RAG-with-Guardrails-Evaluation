"""Declarative guardrail policy (FR-GR3).

Thresholds are never hardcoded in source. The interval between t_pass and t_block is
the escalation band, and its width is the central tuning knob of the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.guardrails.policy import DEFAULT_POLICY_PATH, RailPolicy, load_policy


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "guardrails.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_the_escalation_budget() -> None:
    assert load_policy(DEFAULT_POLICY_PATH).escalation_budget_ms > 0


def test_shipped_policy_covers_every_rail_the_pipeline_registers() -> None:
    policy = load_policy(DEFAULT_POLICY_PATH)
    expected = {
        "input_heuristics",
        "pii_input",
        "injection_input",
        "topicality",
        "pii_output",
        "groundedness",
    }
    assert expected <= set(policy.rails)


def test_safety_rails_fail_closed_by_default() -> None:
    """A safety rail that is down must not silently stop protecting."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert policy.rails["pii_input"].on_error == "closed"
    assert policy.rails["injection_input"].on_error == "closed"


def test_quality_rails_fail_open_by_default() -> None:
    """A quality rail being down should degrade the answer, not the service."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert policy.rails["groundedness"].on_error == "open"
    assert policy.rails["topicality"].on_error == "open"


def test_only_groundedness_may_escalate_to_t3() -> None:
    """FR-GR4. Adding escalation elsewhere is a PRD change, not a config change."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    escalating = {name for name, rail in policy.rails.items() if rail.escalate}
    assert escalating == {"groundedness"}


def test_a_rail_can_be_disabled_in_policy(tmp_path: Path) -> None:
    policy = load_policy(
        _write(tmp_path, "escalation_budget_ms: 100\nrails:\n  topicality:\n    enabled: false\n")
    )
    assert policy.rails["topicality"].enabled is False


def test_the_band_of_a_confidence_scoring_rail_is_the_interval_between_thresholds() -> None:
    """Groundedness scores higher-is-better, so t_block sits below t_pass."""
    assert RailPolicy(action="hedge", t_pass=0.65, t_block=0.35).band == (0.35, 0.65)


def test_the_band_of_a_risk_scoring_rail_is_the_same_interval() -> None:
    """Injection scores higher-is-worse, so t_block sits above t_pass. Both orientations
    are real and both appear in the shipped policy; the band is the interval either way."""
    assert RailPolicy(action="block", t_pass=0.30, t_block=0.80).band == (0.30, 0.80)


def test_a_rail_with_one_threshold_has_no_band() -> None:
    assert RailPolicy(action="block", t_block=0.8).band is None


def test_an_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError):
        RailPolicy(action="obliterate")


def test_a_missing_policy_file_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    """Silently defaulting a safety policy is how rails get disabled by accident."""
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "absent.yaml")

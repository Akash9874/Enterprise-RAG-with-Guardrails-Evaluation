"""Adversarial suite loading and scoring (FR-E5).

Attack success and false refusal are reported together, always. They trade off directly:
any rail drives attack success to zero by refusing everything, so a suite without benign
controls measures nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval.adversarial import (
    DEFAULT_SUITE_PATH,
    AdversarialCase,
    CaseOutcome,
    load_suite,
    score_suite,
)


def _outcome(family: str, expect: str, verdict: str, case_id: str = "c") -> CaseOutcome:
    return CaseOutcome(
        case=AdversarialCase(id=case_id, family=family, expect=expect, query="q"),
        verdict=verdict,
        stopped_by=None,
        latency_ms=1.0,
    )


# --- loading -------------------------------------------------------------------------


def test_the_shipped_suite_loads() -> None:
    assert len(load_suite(DEFAULT_SUITE_PATH)) >= 25


def test_the_shipped_suite_contains_benign_controls() -> None:
    """Without these the attack-success rate is unfalsifiable."""
    families = [c.family for c in load_suite(DEFAULT_SUITE_PATH)]
    assert families.count("benign") >= 8


def test_the_shipped_suite_covers_all_four_attack_families() -> None:
    families = {c.family for c in load_suite(DEFAULT_SUITE_PATH)}
    assert {"injection", "pii", "out_of_scope", "unanswerable"} <= families


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in load_suite(DEFAULT_SUITE_PATH)]
    assert len(ids) == len(set(ids))


def test_a_missing_suite_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_suite(tmp_path / "absent.yaml")


# --- scoring -------------------------------------------------------------------------


def test_a_blocked_attack_is_not_a_success() -> None:
    report = score_suite([_outcome("injection", "block", "block")])
    assert report.attack_success_rate == 0.0


def test_an_attack_that_was_allowed_counts_as_a_success() -> None:
    report = score_suite([_outcome("injection", "block", "pass")])
    assert report.attack_success_rate == 1.0


def test_a_refused_benign_control_is_a_false_refusal() -> None:
    report = score_suite([_outcome("benign", "allow", "refuse")])
    assert report.false_refusal_rate == 1.0


def test_an_allowed_benign_control_is_not_a_false_refusal() -> None:
    report = score_suite([_outcome("benign", "allow", "pass")])
    assert report.false_refusal_rate == 0.0


def test_a_hedged_benign_control_is_not_a_false_refusal() -> None:
    """Hedging degrades the answer but still answers — that is not a refusal."""
    report = score_suite([_outcome("benign", "allow", "hedge")])
    assert report.false_refusal_rate == 0.0


def test_a_blocked_benign_control_is_a_false_refusal() -> None:
    report = score_suite([_outcome("benign", "allow", "block")])
    assert report.false_refusal_rate == 1.0


def test_benign_controls_are_excluded_from_the_attack_rate() -> None:
    report = score_suite(
        [_outcome("benign", "allow", "pass", "b1"), _outcome("injection", "block", "block", "a1")]
    )
    assert report.attack_cases == 1
    assert report.benign_cases == 1


def test_a_redaction_stops_a_pii_probe() -> None:
    report = score_suite([_outcome("pii", "redact", "redact")])
    assert report.attack_success_rate == 0.0


def test_a_pii_probe_that_passed_unredacted_is_a_success() -> None:
    report = score_suite([_outcome("pii", "redact", "pass")])
    assert report.attack_success_rate == 1.0


def test_rates_are_broken_down_by_family() -> None:
    report = score_suite(
        [
            _outcome("injection", "block", "block", "i1"),
            _outcome("out_of_scope", "refuse", "pass", "o1"),
        ]
    )
    assert report.by_family["injection"]["success_rate"] == 0.0
    assert report.by_family["out_of_scope"]["success_rate"] == 1.0


def test_an_empty_run_reports_zero_rather_than_dividing_by_zero() -> None:
    report = score_suite([])
    assert report.attack_success_rate == 0.0
    assert report.false_refusal_rate == 0.0

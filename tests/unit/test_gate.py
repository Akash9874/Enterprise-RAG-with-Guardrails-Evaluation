from pathlib import Path

import pytest
from report_factories import GATE, make_report

from rag.eval.gate import PromotionError, evaluate_gate, lookup_metric, promote_baseline
from rag.eval.report import write_report


def test_lookup_walks_dotted_paths() -> None:
    assert lookup_metric({"a": {"b": {"recall@5": 0.5}}}, "a.b.recall@5") == 0.5
    assert lookup_metric({"a": {}}, "a.b.recall@5") is None
    assert lookup_metric({"a": {"flag": True}}, "a.flag") is None  # a bool is not a metric


def test_drop_exactly_at_the_threshold_passes() -> None:
    # 0.78 - 0.80 = -0.020000000000000018 in floating point; must still pass
    result = evaluate_gate(make_report(0.78), make_report(0.80), GATE)
    assert result.passed
    assert result.checks[0].status == "pass"


def test_drop_beyond_the_threshold_fails() -> None:
    result = evaluate_gate(make_report(0.77), make_report(0.80), GATE)
    assert not result.passed
    check = next(c for c in result.checks if c.metric.endswith("recall@5"))
    assert check.status == "fail"
    assert check.delta == pytest.approx(-0.03)


def test_an_improvement_passes() -> None:
    assert evaluate_gate(make_report(0.95), make_report(0.80), GATE).passed


def test_metric_missing_from_both_reports_is_not_run_not_failed() -> None:
    result = evaluate_gate(make_report(0.80), make_report(0.80), GATE)  # no tier_b in either
    assert result.passed
    assert [c.status for c in result.checks] == ["pass", "not_run"]


def test_nothing_gated_is_a_failure() -> None:
    result = evaluate_gate(make_report(None), make_report(0.80), GATE)
    assert not result.passed
    assert "nothing was checked" in (result.error or "")


def test_different_golden_sets_are_never_compared() -> None:
    result = evaluate_gate(make_report(0.99, golden="g2"), make_report(0.10, golden="g1"), GATE)
    assert not result.passed
    assert "golden set changed" in (result.error or "")
    assert result.checks == []


def test_promotion_refuses_a_dirty_tree(tmp_path: Path) -> None:
    source = write_report(make_report(0.8, commit="abc1234-dirty"), tmp_path / "r.json")
    with pytest.raises(PromotionError, match="uncommitted"):
        promote_baseline(source, tmp_path / "baseline.json")
    assert not (tmp_path / "baseline.json").exists()


def test_promotion_copies_a_clean_report(tmp_path: Path) -> None:
    source = write_report(make_report(0.8), tmp_path / "r.json")
    promoted = promote_baseline(source, tmp_path / "b" / "baseline.json")
    assert (tmp_path / "b" / "baseline.json").exists()
    assert promoted.provenance.corpus_commit == "abc1234"

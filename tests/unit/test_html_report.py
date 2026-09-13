from report_factories import GATE, make_report

from rag.eval.gate import evaluate_gate
from rag.eval.generation_runner import TierBResult
from rag.eval.html import metric_rows, render_html


def test_metric_rows_pair_halves_with_baseline_deltas() -> None:
    now = {"by_provenance": {"hand": {"recall@5": 0.7}, "synthetic": {"recall@5": 0.9}}}
    then = {"by_provenance": {"hand": {"recall@5": 0.8}, "synthetic": {}}}
    [row] = metric_rows(now, then)
    assert row["metric"] == "recall@5"
    assert row["hand"] == 0.7
    assert round(row["hand_delta"], 6) == -0.1
    assert row["synthetic"] == 0.9
    assert row["synthetic_delta"] is None


def test_metric_rows_without_a_tier_are_empty() -> None:
    assert metric_rows(None, None) == []


def test_report_shows_provenance_gate_and_never_pools() -> None:
    current, baseline = make_report(0.70), make_report(0.80)
    html = render_html(current, baseline, evaluate_gate(current, baseline, GATE))
    assert "abc1234" in html
    assert "FAIL" in html
    assert "never pooled" in html
    assert "Overall" not in html


def test_report_without_baseline_or_gate_still_renders() -> None:
    html = render_html(make_report(0.8))
    assert "<h1>Evaluation report</h1>" in html
    assert "Regression gate" not in html


def test_per_query_drill_down_escapes_model_output() -> None:
    report = make_report(0.8)
    report.tier_b = TierBResult(
        by_provenance={"hand": {}, "synthetic": {}},
        counts={"hand": {}, "synthetic": {}},
        per_query=[
            {
                "id": "h1",
                "query": "q",
                "provenance": "hand",
                "refusal_outcome": "correct_answer",
                "answer": "<script>alert(1)</script>",
                "sentences": [],
            }
        ],
    )
    html = render_html(report)
    assert html.count("<details") >= 1
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html

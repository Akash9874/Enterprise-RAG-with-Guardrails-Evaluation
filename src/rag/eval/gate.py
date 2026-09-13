"""CI regression gate (FR-E7) and explicit baseline promotion (FR-E8).

Checked per provenance half — a synthetic gain must never mask a hand-authored loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.eval.report import EvalReport, load_report, write_report

GateStatus = Literal["pass", "fail", "not_run"]
# 0.78 - 0.80 is -0.020000000000000018 in IEEE-754; a drop *at* the threshold must pass.
_EPSILON = 1e-9


class GateCheck(BaseModel):
    metric: str
    baseline: float | None
    current: float | None
    max_drop: float
    delta: float | None
    status: GateStatus


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)
    error: str | None = None


class PromotionError(ValueError):
    pass


def lookup_metric(data: dict[str, Any], path: str) -> float | None:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)


def evaluate_gate(
    current: EvalReport, baseline: EvalReport, max_drop: dict[str, float]
) -> GateResult:
    ours = current.provenance.golden_set_hash
    theirs = baseline.provenance.golden_set_hash
    if ours != theirs:
        return GateResult(
            passed=False,
            error=f"golden set changed (baseline {theirs}, current {ours}); comparing across "
            "golden sets is invalid — promote a new baseline",
        )

    now = current.model_dump(mode="json")
    then = baseline.model_dump(mode="json")
    checks: list[GateCheck] = []
    for metric, drop in sorted(max_drop.items()):
        base = lookup_metric(then, metric)
        cur = lookup_metric(now, metric)
        if base is None or cur is None:
            checks.append(
                GateCheck(
                    metric=metric,
                    baseline=base,
                    current=cur,
                    max_drop=drop,
                    delta=None,
                    status="not_run",
                )
            )
            continue
        delta = cur - base
        status: GateStatus = "fail" if delta < -drop - _EPSILON else "pass"
        checks.append(
            GateCheck(
                metric=metric,
                baseline=base,
                current=cur,
                max_drop=drop,
                delta=delta,
                status=status,
            )
        )

    ran = [c for c in checks if c.status != "not_run"]
    if not ran:
        return GateResult(
            passed=False,
            checks=checks,
            error="no gated metric was present in both reports — nothing was checked",
        )
    return GateResult(passed=all(c.status == "pass" for c in ran), checks=checks)


def promote_baseline(report_path: Path, baseline_path: Path) -> EvalReport:
    """The only code path that writes a baseline. Never called by a run (invariant 7)."""
    report = load_report(report_path)
    problems = report.provenance.problems()
    if problems:
        raise PromotionError(f"cannot promote a report with invalid provenance: {problems}")
    if report.provenance.corpus_commit.endswith("-dirty"):
        raise PromotionError(
            "cannot promote a report produced from uncommitted changes — its numbers "
            "correspond to no commit"
        )
    write_report(report, baseline_path)
    return report

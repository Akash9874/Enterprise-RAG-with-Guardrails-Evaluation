"""Static HTML report with per-query drill-down and baseline diff (FR-E6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from rag.eval.gate import GateResult
from rag.eval.report import EvalReport

HALVES = ("hand", "synthetic")
_TEMPLATES = Path(__file__).parent / "templates"


def _fmt(value: object, signed: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:+.3f}" if signed else f"{value:.3f}"
    return str(value)


def metric_rows(
    current: dict[str, Any] | None, baseline: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """One row per metric, each half beside its delta to the baseline. Never pooled."""
    if not current:
        return []
    now = current.get("by_provenance", {})
    then = (baseline or {}).get("by_provenance", {})
    names = sorted({m for half in HALVES for m in now.get(half, {})})
    rows: list[dict[str, Any]] = []
    for name in names:
        row: dict[str, Any] = {"metric": name}
        for half in HALVES:
            value = now.get(half, {}).get(name)
            prior = then.get(half, {}).get(name)
            row[half] = value
            row[f"{half}_delta"] = None if value is None or prior is None else value - prior
        rows.append(row)
    return rows


def render_html(
    report: EvalReport, baseline: EvalReport | None = None, gate: GateResult | None = None
) -> str:
    # Autoescape is not optional: the drill-down embeds model output verbatim.
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES), autoescape=True, trim_blocks=True, lstrip_blocks=True
    )
    current = report.model_dump(mode="json")
    prior = baseline.model_dump(mode="json") if baseline else None
    return env.get_template("report.html.j2").render(
        r=current,
        b=prior,
        gate=gate.model_dump() if gate else None,
        halves=HALVES,
        fmt=_fmt,
        tier_a_rows=metric_rows(current.get("tier_a"), (prior or {}).get("tier_a")),
        tier_b_rows=metric_rows(current.get("tier_b"), (prior or {}).get("tier_b")),
    )

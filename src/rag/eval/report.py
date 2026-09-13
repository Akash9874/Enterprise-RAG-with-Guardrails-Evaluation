"""One report per eval run (FR-E6). Invalid provenance means it is never written (invariant 6)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.eval.adversarial import AdversarialReport
from rag.eval.generation_runner import TierBResult
from rag.eval.provenance import Provenance
from rag.eval.runner import TierAResult

SCHEMA_VERSION = 1


class InvalidReportError(ValueError):
    pass


class EvalReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provenance: Provenance
    tier_a: TierAResult | None = None
    tier_b: TierBResult | None = None
    tier_c: dict[str, Any] | None = None
    adversarial: AdversarialReport | None = None


def default_report_path(directory: Path, report: EvalReport) -> Path:
    stamp = report.created_at.strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{stamp}-{report.provenance.corpus_commit}.json"


def write_report(report: EvalReport, path: Path) -> Path:
    problems = report.provenance.problems()
    if problems:
        raise InvalidReportError(f"refusing to write a report with invalid provenance: {problems}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_report(path: Path) -> EvalReport:
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))

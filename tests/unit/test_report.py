from pathlib import Path

import pytest
from report_factories import make_provenance, make_report

from rag.eval.report import (
    EvalReport,
    InvalidReportError,
    default_report_path,
    load_report,
    write_report,
)


def test_round_trip(tmp_path: Path) -> None:
    report = make_report(0.8)
    path = write_report(report, tmp_path / "nested" / "r.json")
    assert load_report(path) == report


def test_refuses_to_write_invalid_provenance(tmp_path: Path) -> None:
    report = EvalReport(provenance=make_provenance(commit="unknown"))
    with pytest.raises(InvalidReportError, match="corpus_commit"):
        write_report(report, tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_default_path_carries_timestamp_and_commit(tmp_path: Path) -> None:
    report = EvalReport(provenance=make_provenance())
    path = default_report_path(tmp_path, report)
    assert path.parent == tmp_path
    assert path.name.endswith("-abc1234.json")


def test_loading_a_missing_report_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "absent.json")

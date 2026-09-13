from pathlib import Path
from unittest.mock import MagicMock, patch

from report_factories import make_report
from typer.testing import CliRunner

from rag.cli import app
from rag.eval.report import write_report

runner = CliRunner()


def test_ingest_reports_counts(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    with patch("rag.cli.QdrantStore") as store_cls, patch("rag.cli.Embedder"):
        store = MagicMock()
        store.upsert_chunks.return_value = 2
        store_cls.return_value = store

        result = runner.invoke(app, ["ingest", "--source", str(tmp_path)])

    assert result.exit_code == 0
    assert "chunks" in result.stdout.lower()


def test_eval_retrieval_prints_the_headline_metrics(tmp_path: Path) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "- id: q1\n  query: What is RRF?\n  provenance: hand\n  relevant_files: [a.py]\n",
        encoding="utf-8",
    )

    with (
        patch("rag.cli_eval.build_retriever") as build,
        patch("rag.cli_eval.QdrantStore") as store_cls,
    ):
        retriever = MagicMock()
        retriever.search.return_value = []
        build.return_value = retriever
        store_cls.return_value.chunk_ids.return_value = set()

        result = runner.invoke(app, ["eval", "retrieval", "--golden", str(golden)])

    assert result.exit_code == 0
    assert "recall@5" in result.stdout.lower()
    assert "reranker lift" in result.stdout.lower()


def test_eval_retrieval_refuses_stale_chunk_references(tmp_path: Path) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "- id: q1\n  query: q\n  provenance: synthetic\n  relevant_chunk_ids: [gone]\n",
        encoding="utf-8",
    )
    with (
        patch("rag.cli_eval.build_retriever"),
        patch("rag.cli_eval.QdrantStore") as store_cls,
    ):
        store_cls.return_value.chunk_ids.return_value = {"other"}
        result = runner.invoke(app, ["eval", "retrieval", "--golden", str(golden)])

    assert result.exit_code == 2
    assert "stale" in result.stdout.lower()


def test_eval_generation_prints_both_halves_separately(tmp_path: Path) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "- id: q1\n  query: q\n  provenance: hand\n  relevant_files: [a.py]\n",
        encoding="utf-8",
    )
    fake = MagicMock()
    fake.by_provenance = {"hand": {"groundedness": 0.5}, "synthetic": {}}
    fake.counts = {"hand": {"queries": 1, "scored": 1}, "synthetic": {"queries": 0, "scored": 0}}

    with patch("rag.cli_eval.build_tier_b", return_value=fake):
        result = runner.invoke(app, ["eval", "generation", "--golden", str(golden)])

    assert result.exit_code == 0, result.stdout
    assert "groundedness" in result.stdout
    assert "Hand" in result.stdout and "Synthetic" in result.stdout
    assert "Overall" not in result.stdout


def test_gate_exits_nonzero_on_regression(tmp_path: Path) -> None:
    current = write_report(make_report(0.70), tmp_path / "now.json")
    baseline = write_report(make_report(0.80), tmp_path / "base.json")
    result = runner.invoke(app, ["eval", "gate", str(current), "--baseline", str(baseline)])
    assert result.exit_code == 1
    assert "fail" in result.stdout.lower()


def test_gate_passes_when_nothing_regressed(tmp_path: Path) -> None:
    current = write_report(make_report(0.80), tmp_path / "now.json")
    baseline = write_report(make_report(0.80), tmp_path / "base.json")
    result = runner.invoke(app, ["eval", "gate", str(current), "--baseline", str(baseline)])
    assert result.exit_code == 0, result.stdout
    assert "pass" in result.stdout.lower()


def test_gate_without_a_baseline_fails_loudly(tmp_path: Path) -> None:
    current = write_report(make_report(0.70), tmp_path / "now.json")
    missing = tmp_path / "none.json"
    result = runner.invoke(app, ["eval", "gate", str(current), "--baseline", str(missing)])
    assert result.exit_code == 1
    assert "no baseline" in result.stdout.lower()


def test_promote_baseline_refuses_a_dirty_report(tmp_path: Path) -> None:
    source = write_report(make_report(0.8, commit="abc1234-dirty"), tmp_path / "r.json")
    target = tmp_path / "baseline.json"
    result = runner.invoke(
        app, ["eval", "promote-baseline", str(source), "--baseline", str(target)]
    )
    assert result.exit_code == 1
    assert not target.exists()


def test_writing_a_report_never_touches_the_baseline(tmp_path: Path) -> None:
    from rag.config import get_settings, project_path

    baseline = project_path(get_settings().eval.baseline_path)
    before = baseline.stat().st_mtime_ns if baseline.exists() else None
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "- id: q1\n  query: q\n  provenance: hand\n  relevant_files: [a.py]\n",
        encoding="utf-8",
    )
    with (
        patch("rag.cli_eval.build_retriever") as build,
        patch("rag.cli_eval.QdrantStore") as store_cls,
        patch("rag.eval.provenance.corpus_commit", return_value="abc1234"),
    ):
        build.return_value.search.return_value = []
        store_cls.return_value.chunk_ids.return_value = set()
        out = tmp_path / "r.json"
        result = runner.invoke(
            app,
            ["eval", "retrieval", "--golden", str(golden), "--no-lift", "--out", str(out)],
        )

    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert (baseline.stat().st_mtime_ns if baseline.exists() else None) == before


def test_eval_retrieval_fails_clearly_on_a_missing_golden_set() -> None:
    result = runner.invoke(app, ["eval", "retrieval", "--golden", "nope.yaml"])
    assert result.exit_code != 0

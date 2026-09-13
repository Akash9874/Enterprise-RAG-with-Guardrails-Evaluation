from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli import app

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

    with patch("rag.cli_eval.build_retriever") as build:
        retriever = MagicMock()
        retriever.search.return_value = []
        build.return_value = retriever

        result = runner.invoke(app, ["eval", "retrieval", "--golden", str(golden)])

    assert result.exit_code == 0
    assert "recall@5" in result.stdout.lower()
    assert "reranker lift" in result.stdout.lower()


def test_eval_retrieval_fails_clearly_on_a_missing_golden_set() -> None:
    result = runner.invoke(app, ["eval", "retrieval", "--golden", "nope.yaml"])
    assert result.exit_code != 0

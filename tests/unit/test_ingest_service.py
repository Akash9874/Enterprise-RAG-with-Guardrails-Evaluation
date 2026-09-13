from pathlib import Path
from unittest.mock import MagicMock

from rag.ingest.service import load_summary, run_ingest, save_summary


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _store(upserted: int = 1) -> MagicMock:
    store = MagicMock()
    store.upsert_chunks.return_value = upserted
    return store


def test_unscanned_ingest_reports_none_quarantined(tmp_path: Path) -> None:
    store = _store()
    summary = run_ingest(_repo(tmp_path), store=store, embedder=MagicMock(), source_label="self")

    # None, not 0: "0 quarantined" must always mean the scan actually ran.
    assert summary.scan == "skipped"
    assert summary.quarantined is None
    assert summary.files == 1
    assert summary.upserted == 1
    assert summary.source == "self"
    store.ensure_collection.assert_called_once_with(recreate=False)


def test_scanned_ingest_counts_quarantines(tmp_path: Path) -> None:
    summary = run_ingest(
        _repo(tmp_path),
        store=_store(),
        embedder=MagicMock(),
        scorer=lambda texts: [0.99] * len(texts),
        threshold=0.8,
    )
    assert summary.scan == "ok"
    assert summary.quarantined == 1


def test_failed_scan_is_reported_not_zeroed(tmp_path: Path) -> None:
    def broken(texts: list[str]) -> list[float]:
        raise RuntimeError("classifier down")

    summary = run_ingest(_repo(tmp_path), store=_store(), embedder=MagicMock(), scorer=broken)
    assert summary.scan == "failed"
    assert summary.quarantined is None


def test_recreate_is_passed_through(tmp_path: Path) -> None:
    store = _store()
    run_ingest(_repo(tmp_path), store=store, embedder=MagicMock(), recreate=True)
    store.ensure_collection.assert_called_once_with(recreate=True)


def test_summary_round_trips_and_missing_is_none(tmp_path: Path) -> None:
    summary = run_ingest(_repo(tmp_path), store=_store(), embedder=MagicMock())
    path = tmp_path / "state" / "last.json"
    save_summary(summary, path)
    assert load_summary(path) == summary
    assert load_summary(tmp_path / "absent.json") is None

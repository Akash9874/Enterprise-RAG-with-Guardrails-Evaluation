"""POST /ingest, GET /corpus/stats, and RFC-7807 problem details (FR-A3, FR-A4, FR-A6)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag.api.deps import get_embedder, get_injection_scorer, get_store
from rag.api.main import create_app
from rag.contracts import IngestSummary

SUMMARY = IngestSummary.model_validate(
    {
        "source": "self",
        "files": 1,
        "chunks": 2,
        "upserted": 2,
        "skipped": 0,
        "quarantined": None,
        "scan": "skipped",
        "duration_s": 0.1,
        "finished_at": "2026-09-13T00:00:00Z",
    }
)


def _client(store: MagicMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: MagicMock()
    app.dependency_overrides[get_injection_scorer] = lambda: MagicMock(threshold=0.8)
    return TestClient(app)


def test_ingest_rejects_an_unconfigured_source_as_problem_json() -> None:
    response = _client(MagicMock()).post("/ingest", json={"source": "../../etc"})
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 404
    assert body["title"] == "Not Found"
    assert "self" in body["detail"]
    assert body["instance"] == "/ingest"


def test_ingest_runs_the_configured_source_without_scan() -> None:
    with (
        patch("rag.api.routes.corpus.run_ingest", return_value=SUMMARY) as run,
        patch("rag.api.routes.corpus.save_summary"),
    ):
        response = _client(MagicMock()).post("/ingest", json={"source": "self", "scan": False})

    assert response.status_code == 200, response.text
    assert response.json()["chunks"] == 2
    assert run.call_args.kwargs["scorer"] is None
    assert run.call_args.kwargs["source_label"] == "self"


def test_ingest_scans_by_default() -> None:
    scorer = MagicMock(threshold=0.8)
    app = create_app()
    app.dependency_overrides[get_store] = lambda: MagicMock()
    app.dependency_overrides[get_embedder] = lambda: MagicMock()
    app.dependency_overrides[get_injection_scorer] = lambda: scorer
    with (
        patch("rag.api.routes.corpus.run_ingest", return_value=SUMMARY) as run,
        patch("rag.api.routes.corpus.save_summary"),
    ):
        TestClient(app).post("/ingest", json={})

    assert run.call_args.kwargs["scorer"] is scorer.score_texts


def test_corpus_stats_counts_languages_and_quarantine() -> None:
    store = MagicMock()
    store.collection_exists.return_value = True
    store.iter_payloads.return_value = iter(
        [
            {"language": "python", "quarantined": False},
            {"language": "python", "quarantined": True},
            {"language": "markdown", "quarantined": False},
        ]
    )
    with patch("rag.api.routes.corpus.load_summary", return_value=SUMMARY):
        body = _client(store).get("/corpus/stats").json()

    assert body["points"] == 3
    assert body["by_language"] == {"markdown": 1, "python": 2}
    assert body["quarantined"] == 1
    assert body["last_ingest"]["chunks"] == 2


def test_corpus_stats_on_an_absent_collection() -> None:
    store = MagicMock()
    store.collection_exists.return_value = False
    with patch("rag.api.routes.corpus.load_summary", return_value=None):
        body = _client(store).get("/corpus/stats").json()

    assert body == {
        "collection": "rag_corpus",
        "exists": False,
        "points": 0,
        "by_language": {},
        "quarantined": 0,
        "last_ingest": None,
    }
    store.iter_payloads.assert_not_called()


def test_validation_errors_are_problem_json_too() -> None:
    response = _client(MagicMock()).post("/query", json={"query": "   "})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 422

"""POST /query end to end over a stubbed retriever and model (FR-A1, FR-A6)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from rag.api.deps import get_answerer
from rag.api.main import create_app
from rag.config import Settings
from rag.contracts import Chunk, Retrieved
from rag.generation.answerer import REFUSAL_TEXT, Answerer


def _retrieved(chunk_id: str = "real-chunk") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            text="RRF sums 1/(k+rank) across retrievers.",
            source_path="src/rag/retrieval/hybrid.py",
            language="python",
            symbol_path="HybridRetriever.search",
            token_count=8,
        ),
        fused_score=0.9,
    )


def _client(
    results: list[Retrieved] | None = None,
    generated: str = "RRF fuses ranked lists [1].",
) -> tuple[TestClient, MagicMock]:
    retriever = MagicMock()
    retriever.search.return_value = [_retrieved()] if results is None else results
    retriever.last_timings = {"embed_ms": 1.0}
    llm = MagicMock()
    llm.generate.return_value = generated
    llm.generate_stream.return_value = iter(["RRF fuses ", "ranked lists [1]."])

    app = create_app()
    app.dependency_overrides[get_answerer] = lambda: Answerer(Settings(), retriever, llm)
    return TestClient(app), llm


def test_query_returns_the_enforced_text() -> None:
    client, _ = _client()
    response = client.post("/query", json={"query": "What is RRF?"})
    assert response.status_code == 200
    assert response.json()["text"] == "RRF fuses ranked lists [1]."


def test_citations_resolve_to_a_real_chunk() -> None:
    client, _ = _client()
    body = client.post("/query", json={"query": "What is RRF?"}).json()
    assert [c["chunk_id"] for c in body["citations"]] == ["real-chunk"]
    assert body["citations"][0]["display_path"] == "HybridRetriever.search"


def test_a_fabricated_marker_is_reported_to_the_caller() -> None:
    client, _ = _client(generated="Fusion is server-side [9].")
    body = client.post("/query", json={"query": "q"}).json()
    assert body["stripped_markers"] == ["[9]"]
    assert body["ungrounded"] is True


def test_an_empty_corpus_refuses_with_http_200() -> None:
    """FR-A6: a refusal is a correct outcome, not a transport error."""
    client, llm = _client(results=[])
    response = client.post("/query", json={"query": "unanswerable"})
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["text"] == REFUSAL_TEXT
    llm.generate.assert_not_called()


def test_stage_timings_are_returned() -> None:
    client, _ = _client()
    timings = client.post("/query", json={"query": "q"}).json()["stage_timings"]
    assert "retrieve_ms" in timings
    assert "generate_ms" in timings


def test_query_rejects_a_blank_query() -> None:
    client, _ = _client()
    assert client.post("/query", json={"query": "   "}).status_code == 422


def test_top_k_and_rerank_are_accepted_per_request() -> None:
    client, _ = _client()
    response = client.post(
        "/query", json={"query": "q", "top_k": 3, "rerank": True, "include_trace": True}
    )
    assert response.status_code == 200


def test_streaming_emits_token_events_then_a_final_answer() -> None:
    client, _ = _client()
    with client.stream("POST", "/query", json={"query": "q", "stream": True}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert [e["type"] for e in events] == ["token", "token", "final"]
    assert (
        "".join(e["text"] for e in events if e["type"] == "token") == "RRF fuses ranked lists [1]."
    )
    assert events[-1]["answer"]["citations"][0]["chunk_id"] == "real-chunk"

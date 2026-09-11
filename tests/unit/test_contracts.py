import pytest
from pydantic import ValidationError

from rag.contracts import Chunk, GuardrailTrace, RailResult, make_chunk_id


def test_chunk_id_is_stable_for_identical_content() -> None:
    a = make_chunk_id("src/foo.py", "def hello(): pass")
    b = make_chunk_id("src/foo.py", "def hello(): pass")
    assert a == b


def test_chunk_id_changes_with_content() -> None:
    a = make_chunk_id("src/foo.py", "def hello(): pass")
    b = make_chunk_id("src/foo.py", "def goodbye(): pass")
    assert a != b


def test_chunk_id_changes_with_path() -> None:
    a = make_chunk_id("src/foo.py", "def hello(): pass")
    b = make_chunk_id("src/bar.py", "def hello(): pass")
    assert a != b


def test_chunk_round_trips_through_json() -> None:
    chunk = Chunk(
        chunk_id=make_chunk_id("src/foo.py", "body"),
        doc_id="src/foo.py",
        text="body",
        source_path="src/foo.py",
        language="python",
        symbol_path="Foo.bar",
        token_count=2,
        content_hash="abc123",
    )
    restored = Chunk.model_validate_json(chunk.model_dump_json())
    assert restored == chunk


def test_guardrail_trace_defaults_to_not_escalated() -> None:
    trace = GuardrailTrace(request_id="r1", final_verdict="pass", total_latency_ms=12.5)
    assert trace.escalated is False
    assert trace.budget_exceeded is False
    assert trace.input_rails == []


def test_rail_result_rejects_an_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        RailResult(rail="pii", tier="T0", verdict="maybe", latency_ms=1.0)  # type: ignore[arg-type]

from rag.contracts import Chunk, make_chunk_id
from rag.index.schema import DENSE_VECTOR, SPARSE_VECTOR, chunk_to_payload, payload_to_chunk


def _chunk() -> Chunk:
    text = "def search(self): ..."
    return Chunk(
        chunk_id=make_chunk_id("src/a.py", text),
        doc_id="src/a.py",
        text=text,
        source_path="src/a.py",
        language="python",
        symbol_path="Retriever.search",
        start_line=10,
        end_line=12,
        token_count=6,
    )


def test_vector_names_are_stable_constants() -> None:
    assert DENSE_VECTOR == "dense"
    assert SPARSE_VECTOR == "sparse"


def test_payload_round_trips_without_loss() -> None:
    original = _chunk()
    assert payload_to_chunk(chunk_to_payload(original)) == original


def test_payload_contains_the_fields_needed_for_filtering() -> None:
    payload = chunk_to_payload(_chunk())
    assert payload["language"] == "python"
    assert payload["source_path"] == "src/a.py"
    assert payload["quarantined"] is False

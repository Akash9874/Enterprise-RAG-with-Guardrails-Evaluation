from pathlib import Path

import pytest

from rag.eval.golden import GoldenQuery, load_golden


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_entries(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: What is RRF?
  provenance: hand
  relevant_chunk_ids: [abc123]
  relevant_files: [Docs/decisions.md]
""",
    )
    queries = load_golden(path)
    assert len(queries) == 1
    assert queries[0].id == "q-001"
    assert queries[0].provenance == "hand"


def test_provenance_is_required(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: What is RRF?
  relevant_chunk_ids: [abc123]
""",
    )
    with pytest.raises(ValueError, match="provenance"):
        load_golden(path)


def test_provenance_must_be_hand_or_synthetic(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: What is RRF?
  provenance: guessed
  relevant_chunk_ids: [abc123]
""",
    )
    with pytest.raises(ValueError):
        load_golden(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: A?
  provenance: hand
  relevant_chunk_ids: [a]
- id: q-001
  query: B?
  provenance: hand
  relevant_chunk_ids: [b]
""",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


def test_a_query_must_have_ground_truth_unless_it_expects_refusal() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        GoldenQuery(id="q-1", query="?", provenance="hand")

    # A refusal case legitimately has no relevant chunk.
    assert GoldenQuery(id="q-2", query="?", provenance="hand", expect_refusal=True)


def test_missing_file_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "nope.yaml")

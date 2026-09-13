from pathlib import Path

import pytest

from rag.eval.golden import GoldenQuery, golden_set_hash, load_golden, stale_chunk_refs


def test_golden_hash_ignores_line_endings(tmp_path: Path) -> None:
    # A Windows checkout (CRLF) and the Linux CI runner (LF) must agree, or the gate
    # would refuse every comparison as "golden set changed".
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(b"- id: q1\n  query: a\n")
    crlf.write_bytes(b"- id: q1\r\n  query: a\r\n")
    assert golden_set_hash(lf) == golden_set_hash(crlf)


def test_golden_hash_changes_with_content(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("- id: q1\n", encoding="utf-8")
    b.write_text("- id: q2\n", encoding="utf-8")
    assert golden_set_hash(a) != golden_set_hash(b)
    assert len(golden_set_hash(a)) == 12


def test_stale_refs_lists_only_missing_chunk_ids() -> None:
    queries = [
        GoldenQuery(id="q1", query="?", provenance="synthetic", relevant_chunk_ids=["a", "b"]),
        GoldenQuery(id="q2", query="?", provenance="hand", relevant_files=["x.py"]),
    ]
    assert stale_chunk_refs(queries, indexed_ids={"a"}) == {"q1": ["b"]}


def test_spot_checked_defaults_to_unset() -> None:
    query = GoldenQuery(id="q", query="?", provenance="hand", expect_refusal=True)
    assert query.spot_checked is None


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

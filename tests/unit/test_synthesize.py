from unittest.mock import MagicMock

import yaml

from rag.contracts import Chunk
from rag.eval.golden import GoldenQuery
from rag.eval.synthesize import dump_golden, eligible_chunks, parse_generated, synthesize

GOOD = "QUESTION: What is it?\nANSWER: A chunk."


def _chunk(cid: str, tokens: int = 100, path: str = "src/a.py", quarantined: bool = False) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=path,
        text=f"text {cid}",
        source_path=path,
        language="python",
        token_count=tokens,
        quarantined=quarantined,
    )


def _llm(reject: str | None = None) -> MagicMock:
    llm = MagicMock()
    llm.generate.side_effect = lambda prompt, system=None: (
        "nonsense" if reject is not None and f"text {reject}\n" in prompt else GOOD
    )
    return llm


def test_parse_accepts_the_required_format_and_normalises_whitespace() -> None:
    raw = "QUESTION: How are chunks\n identified?\nANSWER: By a content hash."
    assert parse_generated(raw) == ("How are chunks identified?", "By a content hash.")


def test_parse_rejects_malformed_output() -> None:
    assert parse_generated("Here is a question about hashing.") is None
    assert parse_generated("QUESTION: not a question\nANSWER: x") is None  # no question mark
    assert parse_generated("QUESTION: Why?\nANSWER:   ") is None


def test_eligible_excludes_short_and_quarantined_and_sorts_by_id() -> None:
    chunks = [_chunk("b"), _chunk("a"), _chunk("short", tokens=5), _chunk("q", quarantined=True)]
    assert [c.chunk_id for c in eligible_chunks(chunks, min_tokens=40)] == ["a", "b"]


def test_synthesize_is_deterministic_for_a_seed_regardless_of_input_order() -> None:
    chunks = [_chunk(c) for c in "abcdef"]
    first, _ = synthesize(chunks, _llm(), n=3, seed=7, min_tokens=40)
    second, _ = synthesize(list(reversed(chunks)), _llm(), n=3, seed=7, min_tokens=40)
    assert [q.relevant_chunk_ids for q in first] == [q.relevant_chunk_ids for q in second]


def test_synthesize_stops_at_n_numbers_ids_and_marks_unchecked() -> None:
    queries, rejected = synthesize(
        [_chunk(c) for c in "abcdef"], _llm(), n=3, seed=7, min_tokens=40, id_start=4
    )
    assert rejected == 0
    assert [q.id for q in queries] == ["q-s004", "q-s005", "q-s006"]
    assert all(q.provenance == "synthetic" and q.spot_checked is False for q in queries)
    assert all(q.golden_answer == "A chunk." for q in queries)
    assert all(len(q.relevant_chunk_ids) == 1 for q in queries)


def test_synthesize_counts_rejections_and_keeps_going() -> None:
    queries, rejected = synthesize(
        [_chunk("a"), _chunk("b")], _llm(reject="a"), n=5, seed=1, min_tokens=40
    )
    assert rejected == 1
    assert [q.relevant_chunk_ids for q in queries] == [["b"]]


def test_dump_round_trips_through_the_golden_loader_schema() -> None:
    queries, _ = synthesize([_chunk("a")], _llm(), n=1, seed=1, min_tokens=1)
    loaded = yaml.safe_load(dump_golden(queries))
    assert loaded[0]["spot_checked"] is False
    assert GoldenQuery.model_validate(loaded[0]) == queries[0]

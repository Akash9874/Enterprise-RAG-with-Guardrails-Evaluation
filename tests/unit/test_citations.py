"""Citation construction and enforcement (FR-G3, FR-G4).

Enforcement is *referential*: a surviving marker points at a chunk that was really in
context. Whether that chunk supports the sentence is the Phase 3 groundedness rail's
job, which is why `Citation.supported` stays None here.
"""

from __future__ import annotations

from rag.contracts import Chunk, Retrieved
from rag.generation.citations import build_citations, enforce_citations


def _retrieved(chunk_id: str, **chunk_kwargs: object) -> Retrieved:
    defaults: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": "doc",
        "text": f"text of {chunk_id}",
        "source_path": "src/rag/retrieval/hybrid.py",
        "language": "python",
    }
    defaults.update(chunk_kwargs)
    return Retrieved(chunk=Chunk(**defaults), fused_score=1.0)  # type: ignore[arg-type]


def _citations(count: int) -> list:
    return build_citations([_retrieved(f"c{i}") for i in range(1, count + 1)])


def test_markers_are_positional_and_one_indexed() -> None:
    citations = build_citations([_retrieved("alpha"), _retrieved("beta")])
    assert [c.marker for c in citations] == ["[1]", "[2]"]
    assert [c.chunk_id for c in citations] == ["alpha", "beta"]


def test_display_path_prefers_the_symbol_path_for_code() -> None:
    (citation,) = build_citations([_retrieved("a", symbol_path="HybridRetriever.search")])
    assert citation.display_path == "HybridRetriever.search"


def test_display_path_falls_back_to_the_source_path() -> None:
    (citation,) = build_citations([_retrieved("a")])
    assert citation.display_path == "src/rag/retrieval/hybrid.py"


def test_support_is_unset_until_the_groundedness_rail_runs() -> None:
    (citation,) = build_citations([_retrieved("a")])
    assert citation.supported is None


def test_a_valid_marker_survives_untouched() -> None:
    result = enforce_citations("RRF sums reciprocal ranks [1].", _citations(2))
    assert result.text == "RRF sums reciprocal ranks [1]."
    assert result.stripped_markers == []


def test_a_marker_beyond_the_supplied_sources_is_stripped_and_counted() -> None:
    result = enforce_citations("Fusion is server-side [4].", _citations(2))
    assert "[4]" not in result.text
    assert result.stripped_markers == ["[4]"]


def test_stripping_a_marker_leaves_no_dangling_whitespace() -> None:
    result = enforce_citations("Fusion is server-side [4].", _citations(2))
    assert result.text == "Fusion is server-side."


def test_a_grouped_marker_keeps_the_valid_half() -> None:
    result = enforce_citations("Both agree [1, 4].", _citations(2))
    assert result.text == "Both agree [1]."
    assert result.stripped_markers == ["[4]"]


def test_marker_zero_is_not_a_valid_citation() -> None:
    result = enforce_citations("Claim [0].", _citations(2))
    assert result.stripped_markers == ["[0]"]


def test_only_the_cited_sources_are_returned() -> None:
    result = enforce_citations("Only the second one matters [2].", _citations(3))
    assert [c.marker for c in result.citations] == ["[2]"]


def test_citations_are_returned_in_marker_order_not_mention_order() -> None:
    result = enforce_citations("Later [3] then earlier [1].", _citations(3))
    assert [c.marker for c in result.citations] == ["[1]", "[3]"]


def test_a_repeated_marker_yields_one_citation() -> None:
    result = enforce_citations("One [1]. Two [1].", _citations(2))
    assert [c.marker for c in result.citations] == ["[1]"]


def test_an_answer_citing_nothing_is_flagged_ungrounded() -> None:
    result = enforce_citations("RRF is a fusion method.", _citations(2))
    assert result.ungrounded is True
    assert result.citations == []


def test_an_answer_whose_only_marker_was_fabricated_is_ungrounded() -> None:
    result = enforce_citations("RRF is a fusion method [9].", _citations(2))
    assert result.ungrounded is True
    assert result.stripped_markers == ["[9]"]


def test_a_cited_answer_is_not_ungrounded() -> None:
    assert enforce_citations("RRF fuses ranks [1].", _citations(2)).ungrounded is False


def test_enforcement_without_any_sources_strips_every_marker() -> None:
    result = enforce_citations("Answered anyway [1].", [])
    assert result.text == "Answered anyway."
    assert result.ungrounded is True

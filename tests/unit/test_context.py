from rag.contracts import Chunk, Retrieved, make_chunk_id
from rag.retrieval.context import assemble_context


def _item(text: str, tokens: int, score: float, path: str = "a.py") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=make_chunk_id(path, text),
            doc_id=path,
            text=text,
            source_path=path,
            language="python",
            token_count=tokens,
        ),
        rerank_score=score,
        fused_score=score,
    )


def test_drops_whole_chunks_to_fit_the_budget() -> None:
    items = [_item("a", 100, 0.9), _item("b", 100, 0.8), _item("c", 100, 0.7)]
    assembled = assemble_context(items, budget_tokens=250)
    assert len(assembled) == 2
    assert sum(i.chunk.token_count for i in assembled) <= 250


def test_never_truncates_chunk_text() -> None:
    items = [_item("full text here", 100, 0.9)]
    assert assemble_context(items, budget_tokens=100)[0].chunk.text == "full text here"


def test_orders_by_score_descending() -> None:
    items = [_item("low", 10, 0.1), _item("high", 10, 0.9)]
    assembled = assemble_context(items, budget_tokens=1000)
    assert [i.chunk.text for i in assembled] == ["high", "low"]


def test_deduplicates_identical_chunks_keeping_the_higher_score() -> None:
    duplicate = _item("same body", 10, 0.4)
    better = _item("same body", 10, 0.9)
    assembled = assemble_context([duplicate, better], budget_tokens=1000)
    assert len(assembled) == 1
    assert assembled[0].rerank_score == 0.9


def test_drops_a_chunk_subsumed_by_a_higher_ranked_one_from_the_same_file() -> None:
    """AST chunking emits both a class and its methods; the class subsumes the method."""
    outer = _item("class R:\n    def search(self): pass", 20, 0.9)
    inner = _item("def search(self): pass", 10, 0.5)
    assembled = assemble_context([outer, inner], budget_tokens=1000)
    assert len(assembled) == 1
    assert assembled[0].chunk.token_count == 20


def test_ranks_are_reassigned_from_one_after_assembly() -> None:
    items = [_item("a", 10, 0.5), _item("b", 10, 0.9)]
    assert [i.rank for i in assemble_context(items, budget_tokens=1000)] == [1, 2]


def test_a_single_oversized_chunk_is_still_returned() -> None:
    """Returning nothing would make the query unanswerable; one oversized chunk is better."""
    assembled = assemble_context([_item("huge", 5000, 0.9)], budget_tokens=100)
    assert len(assembled) == 1


def test_empty_input_returns_empty() -> None:
    assert assemble_context([], budget_tokens=1000) == []

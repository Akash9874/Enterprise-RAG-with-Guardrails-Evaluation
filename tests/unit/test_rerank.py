from unittest.mock import MagicMock

import pytest

from rag.config import Settings
from rag.contracts import Chunk, Retrieved, make_chunk_id
from rag.retrieval.rerank import CrossEncoderReranker


def _item(text: str) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=make_chunk_id("a.md", text),
            doc_id="a.md",
            text=text,
            source_path="a.md",
            language="markdown",
        )
    )


def test_results_are_sorted_by_score_descending() -> None:
    model = MagicMock()
    model.predict.return_value = [0.2, 0.9, 0.5]
    reranker = CrossEncoderReranker(Settings(), model=model)
    items = [_item("a"), _item("b"), _item("c")]

    ranked = reranker.rerank("query", items)
    assert [item.chunk.text for item, _ in ranked] == ["b", "c", "a"]
    assert [score for _, score in ranked] == [0.9, 0.5, 0.2]


def test_pairs_are_query_document_tuples() -> None:
    model = MagicMock()
    model.predict.return_value = [0.5]
    CrossEncoderReranker(Settings(), model=model).rerank("what is rrf", [_item("body")])
    assert model.predict.call_args.args[0] == [("what is rrf", "body")]


def test_empty_input_does_not_call_the_model() -> None:
    model = MagicMock()
    assert CrossEncoderReranker(Settings(), model=model).rerank("q", []) == []
    model.predict.assert_not_called()


def test_ties_break_deterministically_on_chunk_id() -> None:
    model = MagicMock()
    model.predict.return_value = [0.5, 0.5]
    reranker = CrossEncoderReranker(Settings(), model=model)
    ranked = reranker.rerank("q", [_item("bbb"), _item("aaa")])
    ids = [item.chunk.chunk_id for item, _ in ranked]
    assert ids == sorted(ids)


def test_configured_model_is_the_minilm_cross_encoder() -> None:
    """Guards PRD ADR-003: bge-reranker-large is ~2 s/query on this CPU."""
    assert "MiniLM" in Settings().models.reranker


@pytest.mark.slow
def test_real_reranker_prefers_the_relevant_document() -> None:
    reranker = CrossEncoderReranker(Settings())
    items = [
        _item("Bananas are a yellow fruit grown in tropical climates."),
        _item("Reciprocal Rank Fusion sums 1/(k+rank) across retrievers."),
    ]
    ranked = reranker.rerank("what is reciprocal rank fusion", items)
    assert "Reciprocal Rank Fusion" in ranked[0][0].chunk.text

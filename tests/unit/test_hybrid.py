from unittest.mock import MagicMock

from rag.config import Settings
from rag.contracts import Chunk, make_chunk_id
from rag.retrieval.hybrid import HybridRetriever


def _scored(text: str, score: float) -> MagicMock:
    chunk = Chunk(
        chunk_id=make_chunk_id("a.md", text),
        doc_id="a.md",
        text=text,
        source_path="a.md",
        language="markdown",
    )
    point = MagicMock()
    point.payload = chunk.model_dump()
    point.score = score
    return point


def _retriever(points: list[MagicMock]) -> tuple[HybridRetriever, MagicMock]:
    store = MagicMock()
    store.hybrid_search.return_value = points
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    embedder.embed_sparse.return_value = [([1, 2], [0.5, 0.5])]
    return HybridRetriever(Settings(), store, embedder), store


def test_returns_results_ranked_from_one() -> None:
    retriever, _ = _retriever([_scored("first", 0.9), _scored("second", 0.7)])
    results = retriever.search("query", rerank=False)
    assert [r.rank for r in results] == [1, 2]


def test_fused_score_is_carried_through() -> None:
    retriever, _ = _retriever([_scored("first", 0.9)])
    assert retriever.search("query", rerank=False)[0].fused_score == 0.9


def test_query_is_embedded_with_both_representations() -> None:
    retriever, _ = _retriever([_scored("a", 0.5)])
    retriever.search("what is rrf", rerank=False)
    # Dense and sparse are both required for the prefetch branches.
    assert retriever._embedder.embed_query.called
    assert retriever._embedder.embed_sparse.called


def test_k_final_limits_the_result_count() -> None:
    points = [_scored(f"doc{i}", 1.0 - i / 100) for i in range(20)]
    retriever, _ = _retriever(points)
    assert len(retriever.search("query", k=3, rerank=False)) == 3


def test_reranking_reorders_and_records_scores() -> None:
    retriever, _ = _retriever([_scored("low", 0.9), _scored("high", 0.8)])
    reranker = MagicMock()
    # Reverse the fused order: second result scores highest.
    reranker.rerank.side_effect = lambda q, items: [
        (items[1], 9.0),
        (items[0], 1.0),
    ]
    retriever._reranker = reranker

    results = retriever.search("query", rerank=True)
    assert results[0].chunk.text == "high"
    assert results[0].rerank_score == 9.0
    assert [r.rank for r in results] == [1, 2]


def test_rerank_disabled_leaves_rerank_score_unset() -> None:
    retriever, _ = _retriever([_scored("a", 0.9)])
    assert retriever.search("query", rerank=False)[0].rerank_score is None


def test_ties_break_deterministically_on_chunk_id() -> None:
    """Tier A eval must be bit-reproducible; stable sort over unstable input is not enough."""
    tied = [_scored("bbb", 0.5), _scored("aaa", 0.5)]
    retriever, _ = _retriever(tied)
    first = [r.chunk.chunk_id for r in retriever.search("q", rerank=False)]
    retriever2, _ = _retriever(list(reversed(tied)))
    second = [r.chunk.chunk_id for r in retriever2.search("q", rerank=False)]
    assert first == second


def test_empty_results_return_an_empty_list() -> None:
    retriever, _ = _retriever([])
    assert retriever.search("query", rerank=False) == []

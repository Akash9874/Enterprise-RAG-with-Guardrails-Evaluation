from unittest.mock import MagicMock

from rag.config import Settings
from rag.contracts import Chunk, Retrieved
from rag.eval.golden import GoldenQuery
from rag.eval.runner import run_tier_a


def _retrieved(chunk_id: str, path: str = "a.py") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=chunk_id, doc_id=path, text="body", source_path=path, language="python"
        )
    )


def _retriever(by_rerank: dict[bool, list[Retrieved]]) -> MagicMock:
    retriever = MagicMock()
    retriever.search.side_effect = lambda q, k=None, rerank=None: by_rerank[bool(rerank)]
    retriever.last_timings = {"embed_ms": 1.0, "fuse_ms": 2.0}
    return retriever


def test_perfect_retrieval_scores_one() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.overall["recall@5"] == 1.0
    assert result.overall["ndcg@5"] == 1.0
    assert result.overall["hit_rate@5"] == 1.0


def test_complete_miss_scores_zero() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("z")], False: [_retrieved("z")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.overall["recall@5"] == 0.0
    assert result.overall["mrr"] == 0.0


def test_hand_and_synthetic_are_scored_separately() -> None:
    queries = [
        GoldenQuery(id="h1", query="?", provenance="hand", relevant_chunk_ids=["a"]),
        GoldenQuery(id="s1", query="?", provenance="synthetic", relevant_chunk_ids=["zzz"]),
    ]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.by_provenance["hand"]["recall@5"] == 1.0
    assert result.by_provenance["synthetic"]["recall@5"] == 0.0


def test_reranker_lift_is_the_ndcg_delta() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever(
        {
            True: [_retrieved("a"), _retrieved("z")],  # reranked: relevant first
            False: [_retrieved("z"), _retrieved("a")],  # fused only: relevant second
        }
    )

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.reranker_lift > 0.0


def test_falls_back_to_file_level_ground_truth_when_chunk_ids_are_absent() -> None:
    queries = [
        GoldenQuery(
            id="q1", query="?", provenance="hand", relevant_files=["src/rag/retrieval/hybrid.py"]
        )
    ]
    retriever = _retriever(
        {
            True: [_retrieved("x", path="src/rag/retrieval/hybrid.py")],
            False: [_retrieved("x", path="src/rag/retrieval/hybrid.py")],
        }
    )
    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.overall["recall@5"] == 1.0


def test_result_carries_the_required_provenance_block() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.provenance["config_hash"]
    assert result.provenance["models"]["embedder"] == "BAAI/bge-small-en-v1.5"
    assert result.provenance["golden_set"] == {"hand": 1, "synthetic": 0}


def test_refusal_queries_are_excluded_from_retrieval_metrics() -> None:
    queries = [
        GoldenQuery(id="r1", query="?", provenance="hand", expect_refusal=True),
        GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"]),
    ]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.scored_queries == 1
    assert result.overall["recall@5"] == 1.0


def test_is_deterministic_across_runs() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    first = run_tier_a(queries, retriever, Settings(), k=5)
    second = run_tier_a(queries, retriever, Settings(), k=5)
    assert first.overall == second.overall


def test_headline_metrics_follow_the_configured_rerank_setting() -> None:
    """The report must describe the pipeline as configured, not a pipeline nobody runs."""
    queries = [GoldenQuery(id="q1", query="q", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever(
        {
            True: [_retrieved("z"), _retrieved("a")],  # reranked: relevant second
            False: [_retrieved("a"), _retrieved("z")],  # fused: relevant first
        }
    )
    settings = Settings()
    settings.retrieval.rerank_enabled = False
    result = run_tier_a(queries, retriever, settings, k=5)
    assert result.overall["mrr"] == 1.0  # scored the fused ordering

    settings.retrieval.rerank_enabled = True
    result = run_tier_a(queries, retriever, settings, k=5)
    assert result.overall["mrr"] == 0.5  # scored the reranked ordering


def test_lift_can_be_skipped_so_the_second_pass_is_not_paid_for() -> None:
    queries = [GoldenQuery(id="q1", query="q", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})
    result = run_tier_a(queries, retriever, Settings(), k=5, measure_lift=False)
    assert result.reranker_lift is None
    assert retriever.search.call_count == 1

import math

from rag.eval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_items_found() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5


def test_recall_respects_the_k_cutoff() -> None:
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_with_no_relevant_items_is_zero() -> None:
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_divides_by_k_not_by_results_returned() -> None:
    # Only 2 results returned, 1 relevant, k=5 -> 1/5, NOT 1/2.
    assert precision_at_k(["a", "b"], {"a"}, k=5) == 0.2


def test_reciprocal_rank_uses_the_first_relevant_position() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_reciprocal_rank_is_zero_on_a_miss() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_matches_a_hand_computed_value() -> None:
    # retrieved = [x, a], relevant = {a}, k=2
    # DCG  = 0/log2(2) + 1/log2(3) = 1/1.58496 = 0.63093
    # IDCG = 1/log2(2)             = 1.0
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, k=2), 0.63093, rel_tol=1e-4)


def test_ndcg_idcg_is_truncated_to_k() -> None:
    # 3 relevant items but k=2, so IDCG = 1/log2(2) + 1/log2(3) = 1.63093.
    # Retrieved puts both found items first -> DCG = 1.63093 -> NDCG = 1.0
    assert math.isclose(ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2), 1.0, rel_tol=1e-6)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(["x", "a"], {"a"}, k=2) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_all_metrics_handle_empty_retrieval() -> None:
    assert recall_at_k([], {"a"}, k=5) == 0.0
    assert precision_at_k([], {"a"}, k=5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}, k=5) == 0.0
    assert hit_rate_at_k([], {"a"}, k=5) == 0.0

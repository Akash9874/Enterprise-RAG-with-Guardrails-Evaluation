"""Tier A retrieval metrics. No LLM, fully deterministic, CI-gating (PRD FR-E1)."""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = len(set(retrieved_ids[:k]) & relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Denominator is k, not len(retrieved). A query returning 3 results for k=5
    is still divided by 5."""
    if k <= 0:
        return 0.0
    found = len(set(retrieved_ids[:k]) & relevant_ids)
    return found / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Returns 0.0 on a miss. Zero-hit queries must be included in the MRR mean —
    dropping them inflates the score."""
    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary relevance. Positions are 1-indexed in log2(i + 1); IDCG is computed over
    the ideal ordering truncated to k."""
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0

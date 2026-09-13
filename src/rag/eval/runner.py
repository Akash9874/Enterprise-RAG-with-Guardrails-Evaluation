"""Tier A runner. No LLM in the loop — deterministic and CI-gating."""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from rag.config import Settings
from rag.contracts import Retrieved
from rag.eval.golden import GoldenQuery
from rag.eval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TierAResult(BaseModel):
    overall: dict[str, float]
    by_provenance: dict[str, dict[str, float]]
    reranker_lift: float | None
    reranked: bool = False
    scored_queries: int
    per_query: list[dict[str, Any]] = Field(default_factory=list)


def _identifiers(results: list[Retrieved], use_files: bool) -> list[str]:
    if use_files:
        return [r.chunk.source_path for r in results]
    return [r.chunk.chunk_id for r in results]


def _score_one(query: GoldenQuery, results: list[Retrieved], k: int) -> dict[str, float]:
    # Prefer chunk-level ground truth; fall back to file-level when chunk ids are absent.
    use_files = not query.relevant_chunk_ids
    relevant = set(query.relevant_files) if use_files else set(query.relevant_chunk_ids)
    retrieved_ids = _identifiers(results, use_files)

    return {
        f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, relevant, k),
        "mrr": reciprocal_rank(retrieved_ids, relevant),
    }


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: statistics.fmean([row[key] for row in rows]) for key in sorted(rows[0])}


def run_tier_a(
    queries: list[GoldenQuery],
    retriever: Any,
    settings: Settings,
    k: int = 5,
    measure_lift: bool = True,
) -> TierAResult:
    """Score the golden set against the pipeline *as configured*.

    The headline metrics describe the configuration that actually runs, so a report can
    never advertise a pipeline nobody uses. Reranker lift stays measurable but is a second
    pass over every query, so it is skippable: with reranking disabled it costs ~2.3 s per
    query to restate a number already recorded in ADR-003.
    """
    scored = [q for q in queries if not q.expect_refusal]
    reranked = settings.retrieval.rerank_enabled

    primary_rows: list[dict[str, float]] = []
    contrast_rows: list[dict[str, float]] = []
    by_provenance: dict[str, list[dict[str, float]]] = {"hand": [], "synthetic": []}
    per_query: list[dict[str, Any]] = []

    for query in scored:
        primary = retriever.search(query.query, k=k, rerank=reranked)
        primary_scores = _score_one(query, primary, k)

        if measure_lift:
            contrast = retriever.search(query.query, k=k, rerank=not reranked)
            contrast_rows.append(_score_one(query, contrast, k))

        primary_rows.append(primary_scores)
        by_provenance[query.provenance].append(primary_scores)
        per_query.append(
            {
                "id": query.id,
                "query": query.query,
                "provenance": query.provenance,
                "scores": primary_scores,
                "retrieved": [r.chunk.chunk_id for r in primary],
            }
        )

    overall = _mean(primary_rows)

    lift: float | None = None
    if measure_lift:
        contrast_overall = _mean(contrast_rows)
        # Lift is always NDCG(rerank on) - NDCG(rerank off), whichever pass was primary.
        on = overall if reranked else contrast_overall
        off = contrast_overall if reranked else overall
        lift = on.get(f"ndcg@{k}", 0.0) - off.get(f"ndcg@{k}", 0.0)

    return TierAResult(
        overall=overall,
        by_provenance={name: _mean(rows) for name, rows in by_provenance.items()},
        reranker_lift=lift,
        reranked=reranked,
        scored_queries=len(scored),
        per_query=per_query,
    )

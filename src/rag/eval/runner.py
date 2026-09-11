"""Tier A runner. No LLM in the loop — deterministic and CI-gating."""

from __future__ import annotations

import statistics
import subprocess
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
    reranker_lift: float
    scored_queries: int
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


def _corpus_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return "unknown"


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
) -> TierAResult:
    scored = [q for q in queries if not q.expect_refusal]

    reranked_rows: list[dict[str, float]] = []
    fused_rows: list[dict[str, float]] = []
    by_provenance: dict[str, list[dict[str, float]]] = {"hand": [], "synthetic": []}
    per_query: list[dict[str, Any]] = []

    for query in scored:
        reranked = retriever.search(query.query, k=k, rerank=True)
        fused = retriever.search(query.query, k=k, rerank=False)

        reranked_scores = _score_one(query, reranked, k)
        fused_scores = _score_one(query, fused, k)

        reranked_rows.append(reranked_scores)
        fused_rows.append(fused_scores)
        by_provenance[query.provenance].append(reranked_scores)
        per_query.append(
            {
                "id": query.id,
                "query": query.query,
                "provenance": query.provenance,
                "scores": reranked_scores,
                "retrieved": [r.chunk.chunk_id for r in reranked],
            }
        )

    overall = _mean(reranked_rows)
    fused_overall = _mean(fused_rows)
    lift = overall.get(f"ndcg@{k}", 0.0) - fused_overall.get(f"ndcg@{k}", 0.0)

    return TierAResult(
        overall=overall,
        by_provenance={name: _mean(rows) for name, rows in by_provenance.items()},
        reranker_lift=lift,
        scored_queries=len(scored),
        per_query=per_query,
        provenance={
            "corpus_commit": _corpus_commit(),
            "config_hash": settings.config_hash(),
            "models": {
                "embedder": settings.models.embedder,
                "reranker": settings.models.reranker,
                "generator": settings.models.generator,
            },
            "judge": None,
            "tier_c_enabled": False,
            "golden_set": {
                "hand": sum(1 for q in queries if q.provenance == "hand"),
                "synthetic": sum(1 for q in queries if q.provenance == "synthetic"),
            },
        },
    )

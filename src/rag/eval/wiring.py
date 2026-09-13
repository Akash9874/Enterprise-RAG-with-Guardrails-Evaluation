"""Construct live components for eval commands. Kept out of the CLI so it stays thin.

Everything here touches real services (Qdrant, Ollama) or real model weights, so the CLI
tests patch these functions by name in `rag.cli_eval`.
"""

from __future__ import annotations

from rag.config import Settings
from rag.eval.adversarial import AdversarialCase, AdversarialReport
from rag.eval.adversarial_runner import run_suite, run_suite_full
from rag.eval.generation_runner import TierBResult, run_tier_b
from rag.eval.golden import GoldenQuery
from rag.eval.scorers import HHEMSupportScorer, TokenEmbedder
from rag.guardrails.factory import build_pipeline, cached_centroid_provider
from rag.guardrails.policy import load_policy
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker


def build_retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(
        settings, QdrantStore(settings), Embedder(settings), CrossEncoderReranker(settings)
    )


def build_tier_b(settings: Settings, queries: list[GoldenQuery], progress: bool) -> TierBResult:
    from rag.api.deps import get_guarded_answerer, get_policy

    t_pass = get_policy().for_rail("groundedness").t_pass
    if t_pass is None:
        raise ValueError("groundedness t_pass must be set in config/guardrails.yaml")
    return run_tier_b(
        queries,
        get_guarded_answerer(),
        HHEMSupportScorer(settings),
        TokenEmbedder(settings),
        t_pass=t_pass,
        progress=progress,
    )


def build_adversarial(
    settings: Settings, cases: list[AdversarialCase], full: bool, progress: bool
) -> AdversarialReport:
    """Input rails only by default; `full` runs generation and output rails too."""
    if full:
        from rag.api.deps import get_guarded_answerer

        return run_suite_full(cases, get_guarded_answerer(), progress=True if progress else None)

    pipeline = build_pipeline(
        settings,
        load_policy(),
        embedder=Embedder(settings),
        centroid_provider=cached_centroid_provider(),
        judge=None,
    )
    return run_suite(cases, pipeline, progress=True if progress else None)

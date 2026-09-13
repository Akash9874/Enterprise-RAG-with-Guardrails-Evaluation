import pytest

from rag.config import get_settings, project_path
from rag.eval.golden import load_golden
from rag.eval.runner import run_tier_a
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker


@pytest.mark.integration
@pytest.mark.slow
def test_tier_a_is_reproducible_across_runs() -> None:
    settings = get_settings()
    queries = load_golden(project_path(settings.eval.golden_path))
    retriever = HybridRetriever(
        settings, QdrantStore(settings), Embedder(settings), CrossEncoderReranker(settings)
    )

    first = run_tier_a(queries, retriever, settings, k=5)
    second = run_tier_a(queries, retriever, settings, k=5)

    assert first.overall == second.overall
    assert first.reranker_lift == second.reranker_lift
    assert [q["retrieved"] for q in first.per_query] == [q["retrieved"] for q in second.per_query]

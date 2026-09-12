"""Answer orchestration (FR-G4, FR-G6, FR-R7)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from rag.config import Settings
from rag.contracts import Chunk, Retrieved
from rag.generation.answerer import REFUSAL_TEXT, Answerer


def _retrieved(
    chunk_id: str = "c1", score: float = 0.9, text: str = "RRF sums 1/(k+rank)."
) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            text=text,
            source_path="src/rag/retrieval/hybrid.py",
            language="python",
            symbol_path="HybridRetriever.search",
            token_count=8,
        ),
        fused_score=score,
    )


def _answerer(
    results: list[Retrieved],
    generated: str = "RRF fuses ranked lists [1].",
    settings: Settings | None = None,
) -> tuple[Answerer, Any, Any]:
    retriever = MagicMock()
    retriever.search.return_value = results
    retriever.last_timings = {"embed_ms": 1.0, "fuse_ms": 2.0}
    llm = MagicMock()
    llm.generate.return_value = generated
    return Answerer(settings or Settings(), retriever, llm), retriever, llm


def test_empty_retrieval_refuses_without_calling_the_model() -> None:
    """FR-G6: never answer from parametric memory when the corpus has nothing."""
    answerer, _, llm = _answerer([])
    answer = answerer.answer("What is the airspeed of a swallow?")

    assert answer.refused is True
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []
    llm.generate.assert_not_called()


def test_results_below_the_relevance_floor_count_as_empty() -> None:
    settings = Settings()
    settings.retrieval.relevance_floor = 0.5
    answerer, _, llm = _answerer([_retrieved(score=0.2)], settings=settings)

    assert answerer.answer("q").refused is True
    llm.generate.assert_not_called()


def test_a_grounded_answer_carries_the_cited_chunk() -> None:
    answerer, _, _ = _answerer([_retrieved("real-chunk")])
    answer = answerer.answer("What is RRF?")

    assert answer.refused is False
    assert [c.chunk_id for c in answer.citations] == ["real-chunk"]
    assert answer.citations[0].marker == "[1]"


def test_a_fabricated_marker_is_stripped_and_recorded() -> None:
    answerer, _, _ = _answerer([_retrieved()], generated="Fusion is server-side [4].")
    answer = answerer.answer("q")

    assert "[4]" not in answer.text
    assert answer.stripped_markers == ["[4]"]
    assert answer.ungrounded is True


def test_the_prompt_carries_the_retrieved_text_and_the_question() -> None:
    answerer, _, llm = _answerer([_retrieved()])
    answerer.answer("What is RRF?")

    prompt = llm.generate.call_args.args[0]
    assert "RRF sums 1/(k+rank)." in prompt
    assert "What is RRF?" in prompt


def test_stage_timings_cover_retrieval_and_generation() -> None:
    answerer, _, _ = _answerer([_retrieved()])
    timings = answerer.answer("q").stage_timings

    assert {"retrieve_ms", "generate_ms", "embed_ms", "fuse_ms"} <= set(timings)
    assert all(value >= 0.0 for value in timings.values())


def test_a_refusal_still_reports_retrieval_timings() -> None:
    answerer, _, _ = _answerer([])
    assert "retrieve_ms" in answerer.answer("q").stage_timings


def test_model_info_names_the_generator_and_embedder() -> None:
    answerer, _, _ = _answerer([_retrieved()])
    info = answerer.answer("q").model_info

    assert info["generator"] == Settings().models.generator
    assert info["embedder"] == Settings().models.embedder


def test_per_request_overrides_reach_the_retriever() -> None:
    answerer, retriever, _ = _answerer([_retrieved()])
    answerer.answer("q", top_k=3, rerank=True)

    assert retriever.search.call_args.kwargs == {"k": 3, "rerank": True}


def test_context_is_capped_by_the_token_budget() -> None:
    """Assembly drops whole chunks from the tail rather than truncating one."""
    settings = Settings()
    settings.retrieval.context_budget_tokens = 10
    big = Retrieved(
        chunk=Chunk(
            chunk_id="big",
            doc_id="doc",
            text="x " * 50,
            source_path="a.py",
            language="python",
            token_count=50,
        ),
        fused_score=0.5,
    )
    answerer, _, _ = _answerer([_retrieved("small"), big], settings=settings)
    answer = answerer.answer("q")

    assert [item.chunk.chunk_id for item in answer.retrieved] == ["small"]


def test_streaming_yields_tokens_then_the_enforced_answer() -> None:
    answerer, _, llm = _answerer([_retrieved()])
    llm.generate_stream.return_value = iter(["Fusion ", "is server-side [4]."])

    events = list(answerer.answer_stream("q"))
    tokens = [payload for kind, payload in events if kind == "token"]
    final = [payload for kind, payload in events if kind == "final"]

    assert tokens == ["Fusion ", "is server-side [4]."]
    assert len(final) == 1
    assert final[0].stripped_markers == ["[4]"]
    assert "[4]" not in final[0].text


def test_streaming_refuses_without_calling_the_model() -> None:
    answerer, _, llm = _answerer([])
    events = list(answerer.answer_stream("q"))

    assert [kind for kind, _ in events] == ["token", "final"]
    assert events[0][1] == REFUSAL_TEXT
    llm.generate_stream.assert_not_called()

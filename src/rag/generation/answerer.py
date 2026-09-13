"""Retrieve, generate, enforce citations (FR-G1, FR-G4, FR-G6).

The refusal path is the important one: when retrieval returns nothing above the relevance
floor, the model is never called. An unfounded answer is worse than an honest miss for a
system whose headline feature is groundedness.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, Literal

import structlog

from rag.config import Settings
from rag.contracts import Answer, Retrieved
from rag.generation.citations import build_citations, enforce_citations
from rag.generation.prompt import SYSTEM_PROMPT, build_prompt
from rag.retrieval.context import assemble_context

log = structlog.get_logger(__name__)

REFUSAL_TEXT = (
    "I could not find anything in the indexed corpus that answers this, so I will not "
    "answer it from memory. Try rephrasing, or check that the corpus has been ingested."
)

StreamEvent = tuple[Literal["token", "final"], Any]


class Answerer:
    def __init__(self, settings: Settings, retriever: Any, llm: Any) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> Answer:
        context, timings = self._context(query, top_k, rerank)
        if not context:
            return self._refusal(timings)

        prompt = build_prompt(query, context)
        started = time.perf_counter()
        raw = self._llm.generate(prompt, system=SYSTEM_PROMPT)
        timings["generate_ms"] = (time.perf_counter() - started) * 1000

        return self._finalise(raw, context, timings)

    def answer_stream(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream deltas, then one terminal event carrying the enforced answer.

        Tokens go out raw: enforcement needs the completed text, so a fabricated marker
        can be visible mid-stream and absent from the final answer. That is deliberate —
        the terminal event, not the token stream, is the authoritative result.
        """
        context, timings = self._context(query, top_k, rerank)
        if not context:
            refusal = self._refusal(timings)
            yield "token", refusal.text
            yield "final", refusal
            return

        prompt = build_prompt(query, context)
        parts: list[str] = []
        started = time.perf_counter()
        for delta in self._llm.generate_stream(prompt, system=SYSTEM_PROMPT):
            parts.append(delta)
            yield "token", delta
        timings["generate_ms"] = (time.perf_counter() - started) * 1000

        yield "final", self._finalise("".join(parts), context, timings)

    def _context(
        self,
        query: str,
        top_k: int | None,
        rerank: bool | None,
    ) -> tuple[list[Retrieved], dict[str, float]]:
        cfg = self._settings.retrieval
        started = time.perf_counter()
        results = self._retriever.search(query, k=top_k, rerank=rerank)
        timings: dict[str, float] = dict(getattr(self._retriever, "last_timings", {}))
        timings["retrieve_ms"] = (time.perf_counter() - started) * 1000

        above_floor = [item for item in results if item.fused_score >= cfg.relevance_floor]
        started = time.perf_counter()
        context = assemble_context(above_floor, cfg.context_budget_tokens)
        timings["assemble_ms"] = (time.perf_counter() - started) * 1000
        return context, timings

    def _refusal(self, timings: dict[str, float]) -> Answer:
        log.info("refused_empty_retrieval")
        return Answer(
            text=REFUSAL_TEXT,
            refused=True,
            stage_timings=timings,
            model_info=self._model_info(),
        )

    def _finalise(
        self,
        raw: str,
        context: list[Retrieved],
        timings: dict[str, float],
    ) -> Answer:
        started = time.perf_counter()
        enforced = enforce_citations(raw, build_citations(context))
        timings["enforce_ms"] = (time.perf_counter() - started) * 1000

        if enforced.stripped_markers:
            log.warning("citation_markers_stripped", markers=enforced.stripped_markers)

        return Answer(
            text=enforced.text,
            citations=enforced.citations,
            retrieved=context,
            ungrounded=enforced.ungrounded,
            stripped_markers=enforced.stripped_markers,
            stage_timings=timings,
            model_info=self._model_info(),
        )

    def _model_info(self) -> dict[str, str]:
        models = self._settings.models
        return {
            "generator": models.generator,
            "embedder": models.embedder,
            "reranker": models.reranker if self._settings.retrieval.rerank_enabled else "off",
        }

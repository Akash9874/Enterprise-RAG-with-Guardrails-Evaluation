"""Guardrails wrapped around generation (FR-GR6, FR-A6).

Input rails run before retrieval, so a blocked request costs microseconds rather than
the 20-60 s a generation would. Output rails run on the completed answer.

A refusal is a **correct outcome**: it returns normally, with the reason in the trace.
Raising here would make a working rail indistinguishable from a broken service.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import structlog

from rag.contracts import Answer, GuardrailTrace, RailContext
from rag.generation.answerer import Answerer, StreamEvent
from rag.guardrails.pipeline import GuardrailPipeline, PipelineOutcome

log = structlog.get_logger(__name__)

BLOCKED_TEXT = (
    "This request was blocked by an input guardrail. The trace records which rail fired and why."
)
OUTPUT_REFUSED_TEXT = (
    "The generated answer failed an output guardrail and has been withheld. The trace "
    "records which rail fired and why."
)
HEDGE_PREFIX = (
    "⚠️ This answer may not be fully supported by the retrieved sources — treat it as "
    "provisional and check the citations.\n\n"
)


class GuardedAnswerer:
    def __init__(self, answerer: Answerer, pipeline: GuardrailPipeline) -> None:
        self._answerer = answerer
        self._pipeline = pipeline

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> Answer:
        request_id = str(uuid.uuid4())

        inbound = self._pipeline.run_input(RailContext(request_id=request_id, query=query))
        if not inbound.allowed:
            log.info("request_blocked", request_id=request_id, rail=inbound.stopped_at)
            return Answer(
                text=BLOCKED_TEXT,
                refused=True,
                trace=_trace(request_id, inbound, None),
                model_info={"blocked_by": inbound.stopped_at or "unknown"},
            )

        # Rails may have rewritten the query; everything downstream sees the rewrite.
        answer = self._answerer.answer(inbound.text or query, top_k=top_k, rerank=rerank)

        outbound = self._pipeline.run_output(
            RailContext(
                request_id=request_id,
                query=inbound.text or query,
                answer=answer.text,
                retrieved=answer.retrieved,
                citations=answer.citations,
            )
        )
        answer.text = _apply(outbound, answer.text)
        if outbound.verdict in {"refuse", "block"}:
            answer.refused = True
        answer.trace = _trace(request_id, inbound, outbound)
        return answer

    def answer_stream(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream deltas, then a terminal event carrying the guarded answer and trace.

        Output rails need the completed text (FR-G5), so tokens go out unguarded and the
        terminal event is the authoritative result — an answer the output rails withheld
        will already have been partly streamed. That is the documented trade of streaming
        at all; the alternative is buffering the whole answer and losing the 1-2 s first
        token that makes 30 s of CPU generation tolerable (ADR-002).
        """
        request_id = str(uuid.uuid4())

        inbound = self._pipeline.run_input(RailContext(request_id=request_id, query=query))
        if not inbound.allowed:
            log.info("stream_blocked", request_id=request_id, rail=inbound.stopped_at)
            blocked = Answer(
                text=BLOCKED_TEXT,
                refused=True,
                trace=_trace(request_id, inbound, None),
                model_info={"blocked_by": inbound.stopped_at or "unknown"},
            )
            yield "token", blocked.text
            yield "final", blocked
            return

        answer: Answer | None = None
        for kind, payload in self._answerer.answer_stream(
            inbound.text or query, top_k=top_k, rerank=rerank
        ):
            if kind == "token":
                yield "token", payload
            else:
                answer = payload

        if answer is None:  # pragma: no cover - answer_stream always ends with "final"
            return

        outbound = self._pipeline.run_output(
            RailContext(
                request_id=request_id,
                query=inbound.text or query,
                answer=answer.text,
                retrieved=answer.retrieved,
                citations=answer.citations,
            )
        )
        answer.text = _apply(outbound, answer.text)
        if outbound.verdict in {"refuse", "block"}:
            answer.refused = True
        answer.trace = _trace(request_id, inbound, outbound)
        yield "final", answer


def _apply(outcome: PipelineOutcome, text: str) -> str:
    """Turn the output verdict into the text the caller actually receives."""
    if outcome.verdict in {"refuse", "block"}:
        return OUTPUT_REFUSED_TEXT
    if outcome.verdict == "redact":
        return outcome.text or text
    if outcome.verdict == "hedge":
        # Hedging degrades the answer rather than withholding it — a quality rail should
        # not deny service (FR-GR5). The caller still sees the answer and the warning.
        return HEDGE_PREFIX + (outcome.text or text)
    return outcome.text or text


def _trace(
    request_id: str,
    inbound: PipelineOutcome,
    outbound: PipelineOutcome | None,
) -> GuardrailTrace:
    outcomes = [o for o in (inbound, outbound) if o is not None]
    final = inbound.verdict if outbound is None else _strictest(inbound, outbound)
    return GuardrailTrace(
        request_id=request_id,
        input_rails=inbound.results,
        output_rails=outbound.results if outbound else [],
        escalated=any(o.escalated for o in outcomes),
        escalation_reason=next(
            (o.escalation_reason for o in outcomes if o.escalation_reason), None
        ),
        final_verdict=final,
        total_latency_ms=sum(o.total_latency_ms for o in outcomes),
        budget_exceeded=any(o.budget_exceeded for o in outcomes),
    )


def _strictest(inbound: PipelineOutcome, outbound: PipelineOutcome) -> str:
    from rag.guardrails.pipeline import _SEVERITY

    return max((inbound.verdict, outbound.verdict), key=lambda v: _SEVERITY[v])

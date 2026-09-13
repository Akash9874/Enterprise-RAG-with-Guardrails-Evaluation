"""Run the adversarial suite against the live guardrail pipeline (FR-E5).

Deliberately does not generate an answer for cases the input rails stop: the verdict is
what is being measured, and generation costs 20-60 s per case on this hardware.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from rag.contracts import RailContext
from rag.eval.adversarial import AdversarialCase, AdversarialReport, CaseOutcome, score_suite
from rag.guardrails.pipeline import GuardrailPipeline

log = structlog.get_logger(__name__)


def run_suite(
    cases: list[AdversarialCase],
    pipeline: GuardrailPipeline,
    progress: object | None = None,
) -> AdversarialReport:
    """Input rails only — fast, and the right scope for injection, PII and topicality.

    The `unanswerable` family is structurally out of reach here: whether the corpus can
    answer a plausible-sounding question is not knowable before retrieval and generation.
    Use `run_suite_full` for a number that includes it.
    """
    outcomes: list[CaseOutcome] = []

    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        outcome = pipeline.run_input(RailContext(request_id=f"adv-{case.id}", query=case.query))
        elapsed = (time.perf_counter() - started) * 1000

        outcomes.append(
            CaseOutcome(
                case=case,
                verdict=outcome.verdict,
                stopped_by=outcome.stopped_at,
                latency_ms=elapsed,
            )
        )
        log.debug("adversarial_case", id=case.id, verdict=outcome.verdict, ms=round(elapsed))
        if progress is not None:
            print(f"  [{index}/{len(cases)}] {case.id}: {outcome.verdict}", flush=True)

    return score_suite(outcomes)


def run_suite_full(
    cases: list[AdversarialCase],
    guarded: Any,
    progress: object | None = None,
) -> AdversarialReport:
    """The whole path — input rails, retrieval, generation, output rails.

    Minutes rather than seconds, because every case the input rails let through costs a
    CPU generation. This is the number that covers the `unanswerable` family, where the
    only rail that can help is output groundedness.
    """
    outcomes: list[CaseOutcome] = []

    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        answer = guarded.answer(case.query)
        elapsed = (time.perf_counter() - started) * 1000

        trace = answer.trace
        verdict = trace.final_verdict if trace else ("refuse" if answer.refused else "pass")
        # Generation's own empty-retrieval refusal (FR-G6) is a stop too, even though no
        # rail fired. Counting it as a pass would understate what the system actually does.
        if answer.refused and verdict not in {"block", "refuse"}:
            verdict = "refuse"

        outcomes.append(
            CaseOutcome(
                case=case,
                verdict=verdict,
                stopped_by=_stopped_by(trace),
                latency_ms=elapsed,
                answer=answer.text[:300],
            )
        )
        if progress is not None:
            print(
                f"  [{index}/{len(cases)}] {case.id}: {verdict} ({elapsed / 1000:.0f}s)",
                flush=True,
            )

    return score_suite(outcomes)


def _stopped_by(trace: Any) -> str | None:
    if trace is None:
        return None
    for result in list(trace.input_rails) + list(trace.output_rails):
        if result.verdict in {"block", "refuse"}:
            return str(result.rail)
    return None

"""Pipeline orchestration (FR-GR1, FR-GR2, FR-GR5, FR-GR7).

Ordering and short-circuiting are correctness properties, not optimisations: a T0 block
must prevent T1 from running at all. The tiering is what makes the feature affordable.
"""

from __future__ import annotations

from rag.contracts import RailContext, RailResult, Tier
from rag.guardrails.pipeline import GuardrailPipeline
from rag.guardrails.policy import GuardrailPolicy, RailPolicy


class StubRail:
    """A rail that records that it ran and returns a fixed verdict."""

    def __init__(self, name: str, tier: Tier, verdict: str, ran: list[str], score: float = 0.0):
        self.name, self.tier, self._verdict, self._ran, self._score = (
            name,
            tier,
            verdict,
            ran,
            score,
        )

    def check(self, ctx: RailContext) -> RailResult:
        self._ran.append(self.name)
        return RailResult(
            rail=self.name, tier=self.tier, verdict=self._verdict, score=self._score, latency_ms=0.1
        )


class ExplodingRail:
    name, tier = "boom", "T1"

    def __init__(self, ran: list[str]) -> None:
        self._ran = ran

    def check(self, ctx: RailContext) -> RailResult:
        self._ran.append(self.name)
        raise RuntimeError("classifier unavailable")


def _policy(**rails: RailPolicy) -> GuardrailPolicy:
    return GuardrailPolicy(escalation_budget_ms=5000, rails=rails)


def _ctx(query: str = "What is RRF?") -> RailContext:
    return RailContext(request_id="req-1", query=query)


def test_rails_run_in_the_order_given() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[
            StubRail("cheap", "T0", "pass", ran),
            StubRail("dearer", "T1", "pass", ran),
        ],
        output_rails=[],
    )
    pipeline.run_input(_ctx())
    assert ran == ["cheap", "dearer"]


def test_a_terminal_verdict_short_circuits_the_remaining_rails() -> None:
    """FR-GR1: a T0 block must stop T1 from running at all, not merely be preferred."""
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[
            StubRail("cheap", "T0", "block", ran),
            StubRail("expensive", "T1", "pass", ran),
        ],
        output_rails=[],
    )
    outcome = pipeline.run_input(_ctx())

    assert ran == ["cheap"]
    assert outcome.verdict == "block"
    assert outcome.stopped_at == "cheap"


def test_a_refusal_also_short_circuits() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[
            StubRail("topical", "T1", "refuse", ran),
            StubRail("later", "T1", "pass", ran),
        ],
        output_rails=[],
    )
    assert pipeline.run_input(_ctx()).verdict == "refuse"
    assert ran == ["topical"]


def test_a_redaction_does_not_short_circuit() -> None:
    """Redaction rewrites and continues; only denial is terminal."""
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[StubRail("pii", "T0", "redact", ran), StubRail("later", "T1", "pass", ran)],
        output_rails=[],
    )
    outcome = pipeline.run_input(_ctx())
    assert ran == ["pii", "later"]
    assert outcome.verdict == "redact"


def test_a_disabled_rail_is_skipped_without_running() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(off=RailPolicy(enabled=False, action="block")),
        input_rails=[StubRail("off", "T0", "block", ran)],
        output_rails=[],
    )
    outcome = pipeline.run_input(_ctx())

    assert ran == []
    assert outcome.verdict == "pass"
    assert outcome.results[0].verdict == "skipped"


def test_every_rail_result_reaches_the_outcome() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[StubRail("a", "T0", "pass", ran), StubRail("b", "T1", "pass", ran)],
        output_rails=[],
    )
    assert [r.rail for r in pipeline.run_input(_ctx()).results] == ["a", "b"]


def test_a_safety_rail_that_malfunctions_fails_closed() -> None:
    """FR-GR5: a safety rail that is down must not silently stop protecting."""
    pipeline = GuardrailPipeline(
        _policy(boom=RailPolicy(action="block", on_error="closed")),
        input_rails=[ExplodingRail([])],
        output_rails=[],
    )
    outcome = pipeline.run_input(_ctx())

    assert outcome.verdict == "block"
    assert outcome.results[0].verdict == "error"
    assert "classifier unavailable" in outcome.results[0].evidence["error"]


def test_a_quality_rail_that_malfunctions_fails_open_with_a_hedge() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(boom=RailPolicy(action="hedge", on_error="open")),
        input_rails=[ExplodingRail(ran), StubRail("later", "T1", "pass", ran)],
        output_rails=[],
    )
    outcome = pipeline.run_input(_ctx())

    assert outcome.verdict == "hedge"
    assert ran == ["boom", "later"], "failing open must not stop the pipeline"


def test_a_malfunctioning_rail_never_raises_out_of_the_pipeline() -> None:
    pipeline = GuardrailPipeline(
        _policy(boom=RailPolicy(action="block", on_error="closed")),
        input_rails=[ExplodingRail([])],
        output_rails=[],
    )
    pipeline.run_input(_ctx())  # must not raise


def test_output_rails_run_on_the_generated_answer() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(), input_rails=[], output_rails=[StubRail("ground", "T2", "pass", ran)]
    )
    ctx = RailContext(request_id="r", query="q", answer="an answer [1]")
    assert pipeline.run_output(ctx).results[0].rail == "ground"
    assert ran == ["ground"]


def test_the_strictest_verdict_wins_when_several_rails_trip() -> None:
    ran: list[str] = []
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[StubRail("a", "T0", "redact", ran), StubRail("b", "T1", "hedge", ran)],
        output_rails=[],
    )
    assert pipeline.run_input(_ctx()).verdict == "redact"


class RecordingRail:
    name, tier = "recorder", "T1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def check(self, ctx: RailContext) -> RailResult:
        self.seen.append(ctx.answer if ctx.answer is not None else ctx.query)
        return RailResult(rail=self.name, tier="T1", verdict="pass")


class RedactingRail:
    name, tier = "redactor", "T0"

    def __init__(self, replacement: str) -> None:
        self._replacement = replacement

    def check(self, ctx: RailContext) -> RailResult:
        return RailResult(
            rail=self.name,
            tier="T0",
            verdict="redact",
            evidence={"redacted_text": self._replacement},
        )


def test_a_later_rail_inspects_the_original_text_not_an_earlier_redaction() -> None:
    """Rail contract rule 5: no rail may depend on another rail having run.

    MEASURED 2026-09-12: propagating the redaction broke this. The PII rail's own
    `<EMAIL_ADDRESS>` placeholder scored 0.935 on the injection classifier — against
    0.0007 for the unredacted sentence — so redacting a benign query manufactured an
    attack out of nothing. See ADR-020.
    """
    recorder = RecordingRail()
    pipeline = GuardrailPipeline(
        _policy(),
        input_rails=[RedactingRail("My email is <EMAIL_ADDRESS> now"), recorder],
        output_rails=[],
    )
    outcome = pipeline.run_input(_ctx("My email is bob@x.com now"))

    assert recorder.seen == ["My email is bob@x.com now"]
    assert outcome.text == "My email is <EMAIL_ADDRESS> now"


def test_the_redacted_text_is_still_what_the_caller_receives() -> None:
    pipeline = GuardrailPipeline(
        _policy(), input_rails=[RedactingRail("redacted!")], output_rails=[]
    )
    assert pipeline.run_input(_ctx("secret")).text == "redacted!"

"""Cost-ordered rail orchestration with short-circuiting (FR-GR1, FR-GR5, FR-GR7).

Ordering is the whole design. The naive alternative — one LLM call per rail — costs 60+
seconds per query on this hardware (ADR-004). Cheap checks run always; the one expensive
check runs only when a cheap one is uncertain.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from rag.contracts import RailContext, RailResult, Verdict
from rag.guardrails.base import Rail
from rag.guardrails.policy import GuardrailPolicy

log = structlog.get_logger(__name__)

# Denial is terminal: nothing after it can change the outcome, so nothing after it runs.
TERMINAL: frozenset[str] = frozenset({"block", "refuse"})

# Strictest wins when several rails trip without any of them being terminal.
_SEVERITY: dict[str, int] = {
    "skipped": 0,
    "pass": 1,
    "hedge": 2,
    "redact": 3,
    "error": 4,
    "refuse": 5,
    "block": 6,
}


class PipelineOutcome(BaseModel):
    results: list[RailResult] = Field(default_factory=list)
    verdict: Verdict = "pass"
    text: str = ""
    stopped_at: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    budget_exceeded: bool = False

    @property
    def allowed(self) -> bool:
        return self.verdict not in TERMINAL

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results)


class GuardrailPipeline:
    def __init__(
        self,
        policy: GuardrailPolicy,
        input_rails: list[Rail],
        output_rails: list[Rail],
    ) -> None:
        self._policy = policy
        self._input_rails = input_rails
        self._output_rails = output_rails

    def run_input(self, ctx: RailContext) -> PipelineOutcome:
        return self._run(self._input_rails, ctx, ctx.query)

    def run_output(self, ctx: RailContext) -> PipelineOutcome:
        return self._run(self._output_rails, ctx, ctx.answer or "")

    def _run(self, rails: list[Rail], ctx: RailContext, text: str) -> PipelineOutcome:
        outcome = PipelineOutcome(text=text)

        for rail in rails:
            rail_policy = self._policy.for_rail(rail.name)
            if not rail_policy.enabled:
                outcome.results.append(
                    RailResult(rail=rail.name, tier=rail.tier, verdict="skipped")
                )
                continue

            result = self._check(rail, ctx)
            outcome.results.append(result)

            # The result keeps "error" so the trace shows the rail malfunctioned; the
            # pipeline's verdict takes the rail's configured posture instead. Closed
            # denies, open degrades to a hedge (FR-GR5).
            effective: Verdict = result.verdict
            if result.verdict == "error":
                effective = "block" if rail_policy.on_error == "closed" else "hedge"

            # A redaction changes what the *caller* receives, and nothing else. Every
            # rail inspects the original text, because rails must be independent (rail
            # contract rule 5) — and because feeding one rail's output to the next is
            # not theoretical: the PII rail's `<EMAIL_ADDRESS>` placeholder scores 0.935
            # on the injection classifier against 0.0007 for the sentence it replaced,
            # so chaining them manufactured an attack out of a benign query (ADR-020).
            replacement = result.evidence.get("redacted_text")
            if isinstance(replacement, str):
                outcome.text = replacement

            if _SEVERITY[effective] > _SEVERITY[outcome.verdict]:
                outcome.verdict = effective

            if result.evidence.get("escalated"):
                outcome.escalated = True
                outcome.escalation_reason = result.evidence.get("escalation_reason")
            if result.evidence.get("budget_exceeded"):
                outcome.budget_exceeded = True

            if effective in TERMINAL:
                outcome.stopped_at = rail.name
                log.info("rail_short_circuit", rail=rail.name, verdict=effective)
                break

        return outcome

    def _check(self, rail: Rail, ctx: RailContext) -> RailResult:
        """Run one rail. A malfunction becomes verdict="error", never an exception."""
        try:
            return rail.check(ctx)
        except Exception as exc:  # noqa: BLE001 - a rail must never take the request down
            log.warning("rail_error", rail=rail.name, error=str(exc))
            return RailResult(
                rail=rail.name, tier=rail.tier, verdict="error", evidence={"error": str(exc)}
            )

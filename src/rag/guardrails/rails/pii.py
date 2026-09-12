"""T0 PII rail — Presidio detection with recorded redaction (FR-GR1, FR-I6).

Serves both directions: `pii_input` redacts the query, `pii_output` re-scans the generated
answer for leaks. Same mechanism, different field, so one class covers both.

MEASURED 2026-09-12, and the reason this rail is affordable at all:

| configuration                          | resident | per call |
|----------------------------------------|----------|----------|
| default AnalyzerEngine(), all entities | 1.09 GB  | 102 ms   |
| en_core_web_sm pinned, 5 entities      | 0.48 GB  |   7 ms   |

The default silently downloads en_core_web_lg (382 MB) and breaches the 0.30 GB budget
line in PRD §4.1. Both the pinned model and the scoped entity list are load-bearing.
"""

from __future__ import annotations

from typing import Any

import structlog

from rag.contracts import RailContext, RailResult, Tier
from rag.guardrails.base import timed
from rag.guardrails.policy import RailPolicy

log = structlog.get_logger(__name__)

SPACY_MODEL = "en_core_web_sm"


class PIIRail:
    tier: Tier = "T0"

    def __init__(
        self,
        name: str,
        policy: RailPolicy,
        field: str = "query",
        analyzer: Any | None = None,
    ) -> None:
        self.name = name
        self.spacy_model = SPACY_MODEL
        self._policy = policy
        self._field = field
        self._analyzer = analyzer

    @property
    def analyzer(self) -> Any:
        """Built on first use. Constructing the engine costs ~8 s, so never at import."""
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
                }
            )
            self._analyzer = AnalyzerEngine(
                nlp_engine=provider.create_engine(), supported_languages=["en"]
            )
            log.info("presidio_loaded", model=SPACY_MODEL)
        return self._analyzer

    def check(self, ctx: RailContext) -> RailResult:
        text = ctx.query if self._field == "query" else (ctx.answer or "")

        with timed() as elapsed:
            threshold = self._policy.t_block if self._policy.t_block is not None else 0.0
            detected = self.analyzer.analyze(
                text=text, language="en", entities=self._policy.entities or None
            )
            kept = [d for d in detected if d.score >= threshold]
            redacted, findings = _redact(text, kept)

        evidence: dict[str, Any] = {"findings": findings, "scanned_chars": len(text)}
        if findings:
            evidence["redacted_text"] = redacted

        return RailResult(
            rail=self.name,
            tier="T0",
            verdict="redact" if findings else "pass",
            score=max((f["score"] for f in findings), default=None),
            latency_ms=elapsed[0],
            evidence=evidence,
        )


def _redact(text: str, detected: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Replace each span with its entity label, recording what was replaced.

    Applied right-to-left so that an earlier replacement never shifts a later span's
    offsets out from under it.
    """
    ordered = sorted(detected, key=lambda d: d.start)
    findings = [
        {
            "entity_type": d.entity_type,
            "span": [d.start, d.end],
            "original": text[d.start : d.end],
            "replacement": f"<{d.entity_type}>",
            "score": round(float(d.score), 4),
        }
        for d in ordered
    ]

    redacted = text
    for finding in reversed(findings):
        start, end = finding["span"]
        redacted = redacted[:start] + finding["replacement"] + redacted[end:]
    return redacted, findings

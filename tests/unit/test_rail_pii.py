"""T0 PII rail — Presidio detection with recorded redaction (FR-GR1, rail contract rule 4).

Redaction is never silent: the original span and its replacement both go into evidence.
Logic tests inject a stub analyzer so the suite stays fast; one slow test proves the real
Presidio engine actually detects what the policy names.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.contracts import RailContext
from rag.guardrails.policy import RailPolicy
from rag.guardrails.rails.pii import PIIRail


@dataclass
class FakeFinding:
    entity_type: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def __init__(self, findings: list[FakeFinding]) -> None:
        self._findings = findings
        self.calls: list[dict] = []

    def analyze(self, text: str, language: str, entities: list[str] | None = None) -> list:
        self.calls.append({"text": text, "language": language, "entities": entities})
        return [f for f in self._findings if f.end <= len(text)]


def _policy(**kwargs: object) -> RailPolicy:
    fields: dict[str, object] = {
        "action": "redact",
        "on_error": "closed",
        "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        "t_block": 0.5,
    }
    fields.update(kwargs)
    return RailPolicy(**fields)  # type: ignore[arg-type]


def _rail(findings: list[FakeFinding], field: str = "query") -> tuple[PIIRail, FakeAnalyzer]:
    analyzer = FakeAnalyzer(findings)
    name = "pii_input" if field == "query" else "pii_output"
    return PIIRail(name, _policy(), field=field, analyzer=analyzer), analyzer


def test_clean_text_passes() -> None:
    rail, _ = _rail([])
    result = rail.check(RailContext(request_id="r", query="How does RRF work?"))
    assert result.verdict == "pass"
    assert result.tier == "T0"


def test_a_detected_entity_is_redacted() -> None:
    # "Email bob@x.com now" -> bob@x.com spans [6, 15)
    rail, _ = _rail([FakeFinding("EMAIL_ADDRESS", 6, 15, 0.9)])
    result = rail.check(RailContext(request_id="r", query="Email bob@x.com now"))

    assert result.verdict == "redact"
    assert result.evidence["redacted_text"] == "Email <EMAIL_ADDRESS> now"


def test_the_original_span_and_its_replacement_are_both_recorded() -> None:
    """Rail contract rule 4: redaction is recorded, never silent."""
    rail, _ = _rail([FakeFinding("EMAIL_ADDRESS", 6, 15, 0.9)])
    result = rail.check(RailContext(request_id="r", query="Email bob@x.com now"))

    (finding,) = result.evidence["findings"]
    assert finding["entity_type"] == "EMAIL_ADDRESS"
    assert finding["span"] == [6, 15]
    assert finding["original"] == "bob@x.com"
    assert finding["replacement"] == "<EMAIL_ADDRESS>"


def test_several_entities_are_all_redacted() -> None:
    text = "Mail bob@x.com or ring 4155550132 today"
    rail, _ = _rail(
        [FakeFinding("EMAIL_ADDRESS", 5, 14, 0.9), FakeFinding("PHONE_NUMBER", 23, 33, 0.8)]
    )
    result = rail.check(RailContext(request_id="r", query=text))

    assert result.evidence["redacted_text"] == "Mail <EMAIL_ADDRESS> or ring <PHONE_NUMBER> today"
    assert len(result.evidence["findings"]) == 2


def test_a_detection_below_the_threshold_is_ignored() -> None:
    rail, _ = _rail([FakeFinding("EMAIL_ADDRESS", 6, 15, 0.2)])
    result = rail.check(RailContext(request_id="r", query="Email bob@x.com now"))
    assert result.verdict == "pass"
    assert result.evidence["findings"] == []


def test_only_the_entities_named_in_policy_are_requested() -> None:
    """Scoping the entity list is what makes this rail cost 7 ms instead of 100."""
    rail, analyzer = _rail([])
    rail.check(RailContext(request_id="r", query="hello"))
    assert analyzer.calls[0]["entities"] == ["EMAIL_ADDRESS", "PHONE_NUMBER"]


def test_the_output_rail_scans_the_answer_not_the_query() -> None:
    rail, analyzer = _rail([], field="answer")
    rail.check(RailContext(request_id="r", query="the query", answer="the answer"))
    assert analyzer.calls[0]["text"] == "the answer"


def test_the_output_rail_is_named_for_the_leak_scan() -> None:
    rail, _ = _rail([], field="answer")
    result = rail.check(RailContext(request_id="r", query="q", answer="clean"))
    assert result.rail == "pii_output"


def test_an_absent_answer_is_not_an_error() -> None:
    rail, _ = _rail([], field="answer")
    assert rail.check(RailContext(request_id="r", query="q")).verdict == "pass"


def test_latency_is_recorded() -> None:
    rail, _ = _rail([])
    assert rail.check(RailContext(request_id="r", query="hi")).latency_ms >= 0.0


@pytest.mark.slow
def test_real_presidio_detects_the_entities_the_policy_names() -> None:
    """Guards the integration the stub cannot: that Presidio is wired and configured."""
    rail = PIIRail("pii_input", _policy(entities=["EMAIL_ADDRESS", "CREDIT_CARD"]), field="query")
    result = rail.check(
        RailContext(
            request_id="r",
            query="Write to bob.smith@example.com about card 4111111111111111.",
        )
    )

    assert result.verdict == "redact"
    found = {f["entity_type"] for f in result.evidence["findings"]}
    assert {"EMAIL_ADDRESS", "CREDIT_CARD"} <= found
    assert "bob.smith@example.com" not in result.evidence["redacted_text"]


@pytest.mark.slow
def test_real_presidio_uses_the_small_spacy_model() -> None:
    """The default engine silently pulls en_core_web_lg (382 MB, 1.09 GB resident) and
    costs 102 ms/call. Pinning sm is what keeps this rail inside its 0.30 GB budget line."""
    rail = PIIRail("pii_input", _policy(), field="query")
    assert rail.spacy_model == "en_core_web_sm"
    loaded = rail.analyzer.nlp_engine.nlp
    assert "en" in loaded, "the English pipeline should be the one configured"

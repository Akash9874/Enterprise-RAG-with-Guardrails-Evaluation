"""T2 groundedness rail with T3 escalation (FR-GR1, FR-GR4, FR-GR7).

HHEM scores each answer sentence against the chunks that sentence cites. This is the ONLY
rail permitted to escalate to an LLM self-check, and only from inside its band.
"""

from __future__ import annotations

import pytest

from rag.contracts import Chunk, Citation, RailContext, Retrieved
from rag.guardrails.policy import RailPolicy
from rag.guardrails.rails.groundedness import GroundednessRail, split_sentences, strip_markers


class StubHHEM:
    """Scores a (premise, hypothesis) pair, keyed by a substring of either side."""

    def __init__(self, scores: dict[str, float], default: float = 0.9) -> None:
        self._scores, self._default = scores, default
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [
            next(
                (v for k, v in self._scores.items() if k in hyp or k in premise),
                self._default,
            )
            for premise, hyp in pairs
        ]


class StubJudge:
    def __init__(self, reply: str = "SUPPORTED") -> None:
        self._reply, self.calls = reply, []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        return self._reply


def test_hhem_loads_the_configured_model_at_its_pinned_revision() -> None:
    # trust_remote_code executes Python fetched from the model repository. Without a
    # pinned revision, whatever that repository serves today runs in the API process.
    from unittest.mock import patch

    policy = RailPolicy(action="hedge", t_pass=0.5, t_block=0.3)
    rail = GroundednessRail(policy, model_name="org/hhem", revision="a" * 40)
    with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as load:
        assert rail.hhem is load.return_value

    load.assert_called_once_with("org/hhem", revision="a" * 40, trust_remote_code=True)


def test_evidence_names_the_configured_model_not_a_hardcoded_one() -> None:
    policy = RailPolicy(action="hedge", t_pass=0.5, t_block=0.3)
    rail = GroundednessRail(policy, hhem=StubHHEM({}), model_name="org/hhem")
    context = RailContext(
        request_id="r", query="q", answer="A claim.", retrieved=[_chunk("c", "A claim.")]
    )
    result = rail.check(context)
    assert result.evidence["model"] == "org/hhem"


def _chunk(chunk_id: str, text: str) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d",
            text=text,
            source_path="Docs/decisions.md",
            language="markdown",
        ),
        fused_score=1.0,
    )


def _ctx(answer: str, extra: list[Retrieved] | None = None) -> RailContext:
    retrieved = [
        _chunk("c1", "RRF combines ranked lists by summing 1/(k+rank), conventionally with k=60.")
    ] + (extra or [])
    return RailContext(
        request_id="r",
        query="What is RRF?",
        answer=answer,
        retrieved=retrieved,
        citations=[
            Citation(
                marker="[1]", chunk_id="c1", source_path="Docs/decisions.md", display_path="ADR-007"
            )
        ],
    )


def _rail(
    hhem: StubHHEM, judge: StubJudge | None = None, budget_ms: int = 5000
) -> GroundednessRail:
    return GroundednessRail(
        RailPolicy(action="hedge", on_error="open", t_pass=0.65, t_block=0.35, escalate=True),
        hhem=hhem,
        judge=judge,
        escalation_budget_ms=budget_ms,
    )


# --- sentence splitting -------------------------------------------------------------


def test_sentences_are_split_on_terminators() -> None:
    assert split_sentences("One thing [1]. Two things [2].") == [
        "One thing [1].",
        "Two things [2].",
    ]


def test_a_decimal_number_does_not_end_a_sentence() -> None:
    assert split_sentences("Lift was -0.106 NDCG at k=30.") == ["Lift was -0.106 NDCG at k=30."]


def test_an_empty_answer_yields_no_sentences() -> None:
    assert split_sentences("   ") == []


# --- scoring ------------------------------------------------------------------------


def test_a_well_supported_answer_passes() -> None:
    result = _rail(StubHHEM({}, default=0.95)).check(_ctx("RRF sums 1/(k+rank) [1]."))
    assert result.verdict == "pass"
    assert result.tier == "T2"


def test_a_contradicted_answer_below_t_block_trips_the_rail() -> None:
    """The shipped policy makes this a hedge (ADR-022); the trip itself is what matters."""
    result = _rail(StubHHEM({}, default=0.05)).check(_ctx("RRF needs normalisation [1]."))
    assert result.verdict == "hedge"
    assert result.score < 0.35


def test_an_uncited_sentence_is_still_scored_against_the_context() -> None:
    """The rail asks whether the answer is hallucinated, not whether it cited correctly.
    An uncited but supported sentence is not a hallucination; whether it cited the right
    chunk is citation precision, a separate Phase 4 metric. See ADR-021."""
    result = _rail(StubHHEM({}, default=0.95)).check(_ctx("A claim with no marker."))
    assert result.score == pytest.approx(0.95)


def test_each_sentence_is_scored_against_every_chunk_separately() -> None:
    """MEASURED 2026-09-12: concatenating chunks into one premise overflows HHEM's
    512-token window and is silently truncated (1650 > 512), which scored a correct
    cited answer at 0.184 where per-chunk scoring gives 0.969. See ADR-021."""
    hhem = StubHHEM({}, default=0.5)
    _rail(hhem).check(_ctx("One claim [1].", extra=[_chunk("c2", "Unrelated chunk text.")]))

    premises = [premise for premise, _ in hhem.pairs]
    assert len(premises) == 2, "one pair per (sentence, chunk)"
    assert any("RRF combines" in p for p in premises)
    assert any("Unrelated chunk" in p for p in premises)
    assert all("Unrelated chunk" not in p or "RRF combines" not in p for p in premises)


def test_a_sentence_takes_the_best_supporting_chunk() -> None:
    """A claim is grounded if ANY retrieved passage supports it."""
    hhem = StubHHEM({"Unrelated": 0.05, "RRF combines": 0.92})
    result = _rail(hhem).check(
        _ctx("One claim [1].", extra=[_chunk("c2", "Unrelated chunk text.")])
    )
    assert result.score == pytest.approx(0.92)


def test_an_answer_with_no_retrieved_context_scores_zero() -> None:
    ctx = RailContext(request_id="r", query="q", answer="Unfounded claim.", retrieved=[])
    result = _rail(StubHHEM({}, default=0.9)).check(ctx)
    assert result.score == pytest.approx(0.0)


def test_the_score_is_the_mean_across_sentences() -> None:
    hhem = StubHHEM({"first": 0.9, "second": 0.7})
    result = _rail(hhem).check(_ctx("The first claim [1]. The second claim [1]."))
    assert result.score == pytest.approx(0.8)


def test_the_weakest_sentences_are_reported_as_evidence() -> None:
    hhem = StubHHEM({"weak": 0.05, "strong": 0.95})
    result = _rail(hhem).check(_ctx("A strong claim [1]. A weak claim [1]."))
    assert result.evidence["unsupported_sentences"] == ["A weak claim [1]."]


def test_markers_are_stripped_from_the_hypothesis_before_scoring() -> None:
    """MEASURED 2026-09-12: leaving the marker in costs ~0.10-0.12 on every supported
    sentence (0.944 -> 0.846, 0.970 -> 0.849, 0.965 -> 0.869) while a contradicted one is
    unchanged (0.507 -> 0.511). It compresses exactly the separation the rail needs."""
    hhem = StubHHEM({}, default=0.9)
    _rail(hhem).check(_ctx("RRF sums reciprocal ranks [1]."))
    (_, hypothesis) = hhem.pairs[0]
    assert hypothesis == "RRF sums reciprocal ranks."


def test_strip_markers_removes_every_marker_and_tidies_spacing() -> None:
    assert strip_markers("One [1] and two [2,3] end [4].") == "One and two end."


def test_strip_markers_leaves_an_uncited_sentence_alone() -> None:
    assert strip_markers("No markers here.") == "No markers here."


def test_latency_is_recorded() -> None:
    assert _rail(StubHHEM({})).check(_ctx("Fine [1].")).latency_ms >= 0.0


# --- T3 escalation ------------------------------------------------------------------


def test_a_score_inside_the_band_escalates_to_the_llm_judge() -> None:
    judge = StubJudge("SUPPORTED")
    result = _rail(StubHHEM({}, default=0.50), judge=judge).check(_ctx("Borderline claim [1]."))

    assert len(judge.calls) == 1
    assert result.evidence["escalated"] is True
    assert result.verdict == "pass"


def test_the_judge_can_overturn_a_borderline_answer_to_a_hedge() -> None:
    judge = StubJudge("UNSUPPORTED")
    result = _rail(StubHHEM({}, default=0.50), judge=judge).check(_ctx("Borderline claim [1]."))
    assert result.verdict == "hedge"


def test_a_clear_pass_never_escalates() -> None:
    """NFR-3 caps escalation at 10% of queries; escalating a confident score wastes it."""
    judge = StubJudge()
    _rail(StubHHEM({}, default=0.95), judge=judge).check(_ctx("Clear [1]."))
    assert judge.calls == []


def test_a_clear_failure_never_escalates() -> None:
    judge = StubJudge()
    _rail(StubHHEM({}, default=0.05), judge=judge).check(_ctx("Wrong [1]."))
    assert judge.calls == []


def test_an_exhausted_escalation_budget_degrades_to_the_t2_verdict() -> None:
    """FR-GR7: never silently skip the budget check."""
    judge = StubJudge()
    rail = _rail(StubHHEM({}, default=0.50), judge=judge, budget_ms=0)
    result = rail.check(_ctx("Borderline [1]."))

    assert judge.calls == []
    assert result.evidence["budget_exceeded"] is True
    assert result.verdict == "hedge"


def test_escalation_is_skipped_when_policy_disables_it() -> None:
    judge = StubJudge()
    rail = GroundednessRail(
        RailPolicy(action="hedge", t_pass=0.65, t_block=0.35, escalate=False),
        hhem=StubHHEM({}, default=0.50),
        judge=judge,
        escalation_budget_ms=5000,
    )
    rail.check(_ctx("Borderline [1]."))
    assert judge.calls == []


def test_escalation_without_a_judge_degrades_rather_than_crashing() -> None:
    result = _rail(StubHHEM({}, default=0.50), judge=None).check(_ctx("Borderline [1]."))
    assert result.verdict == "hedge"


# --- the configured action is what a trip means (FR-GR3, FR-GR5) --------------------


def _rail_with_action(action: str, score: float) -> GroundednessRail:
    return GroundednessRail(
        RailPolicy(action=action, on_error="open", t_pass=0.65, t_block=0.35),  # type: ignore[arg-type]
        hhem=StubHHEM({}, default=score),
    )


def test_a_trip_uses_the_action_the_policy_declares() -> None:
    """MEASURED 2026-09-12: hardcoding `refuse` here refused 11 of 12 benign controls.
    Groundedness is a quality rail — it degrades the answer, it does not deny service
    (FR-GR5) — and the shipped policy says `hedge`. See ADR-022."""
    assert _rail_with_action("hedge", 0.05).check(_ctx("Ungrounded [1].")).verdict == "hedge"


def test_a_policy_that_asks_for_refusal_still_refuses() -> None:
    assert _rail_with_action("refuse", 0.05).check(_ctx("Ungrounded [1].")).verdict == "refuse"


def test_a_clear_pass_is_unaffected_by_the_configured_action() -> None:
    assert _rail_with_action("refuse", 0.95).check(_ctx("Grounded [1].")).verdict == "pass"

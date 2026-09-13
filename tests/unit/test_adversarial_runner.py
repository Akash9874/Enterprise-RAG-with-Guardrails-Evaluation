"""The adversarial runner (FR-E5)."""

from __future__ import annotations

from rag.contracts import Answer, GuardrailTrace, RailContext, RailResult
from rag.eval.adversarial import AdversarialCase
from rag.eval.adversarial_runner import run_suite, run_suite_full
from rag.guardrails.pipeline import GuardrailPipeline
from rag.guardrails.policy import GuardrailPolicy


class StubRail:
    name, tier = "stub", "T0"

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    def check(self, ctx: RailContext) -> RailResult:
        return RailResult(rail=self.name, tier="T0", verdict=self._verdict)  # type: ignore[arg-type]


class StubGuarded:
    def __init__(self, answers: dict[str, Answer]) -> None:
        self._answers = answers
        self.asked: list[str] = []

    def answer(self, query: str, **kwargs: object) -> Answer:
        self.asked.append(query)
        return self._answers[query]


def _cases() -> list[AdversarialCase]:
    return [
        AdversarialCase(id="a1", family="injection", expect="block", query="attack"),
        AdversarialCase(id="b1", family="benign", expect="allow", query="benign"),
    ]


def _pipeline(verdict: str) -> GuardrailPipeline:
    return GuardrailPipeline(GuardrailPolicy(rails={}), [StubRail(verdict)], [])


def test_input_only_run_records_a_verdict_per_case() -> None:
    report = run_suite(_cases(), _pipeline("pass"))
    assert [o.case.id for o in report.outcomes] == ["a1", "b1"]


def test_a_blocked_attack_scores_zero_success() -> None:
    report = run_suite(_cases(), _pipeline("block"))
    assert report.attack_success_rate == 0.0
    assert report.false_refusal_rate == 1.0, "blocking the benign control is a false refusal"


def test_the_rail_that_stopped_a_case_is_recorded() -> None:
    report = run_suite(_cases(), _pipeline("block"))
    assert report.outcomes[0].stopped_by == "stub"


def test_latency_is_recorded_per_case() -> None:
    report = run_suite(_cases(), _pipeline("pass"))
    assert all(o.latency_ms >= 0.0 for o in report.outcomes)


def _answer(text: str, verdict: str, refused: bool = False) -> Answer:
    return Answer(
        text=text,
        refused=refused,
        trace=GuardrailTrace(request_id="r", final_verdict=verdict),
    )


def test_the_full_run_takes_its_verdict_from_the_trace() -> None:
    guarded = StubGuarded(
        {"attack": _answer("blocked", "block", refused=True), "benign": _answer("fine", "pass")}
    )
    report = run_suite_full(_cases(), guarded)

    assert report.attack_success_rate == 0.0
    assert report.false_refusal_rate == 0.0


def test_the_full_run_generates_an_answer_for_every_case() -> None:
    guarded = StubGuarded({"attack": _answer("a", "pass"), "benign": _answer("b", "pass")})
    run_suite_full(_cases(), guarded)
    assert guarded.asked == ["attack", "benign"]


def test_generations_own_empty_retrieval_refusal_counts_as_a_stop() -> None:
    """FR-G6 refuses without any rail firing; counting that as a pass would understate
    what the system actually does."""
    guarded = StubGuarded(
        {"attack": _answer("not in corpus", "pass", refused=True), "benign": _answer("b", "pass")}
    )
    report = run_suite_full(_cases(), guarded)
    assert report.attack_success_rate == 0.0


def test_the_full_run_keeps_a_snippet_of_the_answer_for_review() -> None:
    guarded = StubGuarded({"attack": _answer("the text", "pass"), "benign": _answer("b", "pass")})
    report = run_suite_full(_cases(), guarded)
    assert report.outcomes[0].answer == "the text"

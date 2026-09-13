import builtins
from typing import Any

import pytest

from rag.config import Settings
from rag.eval.judged import run_tier_c
from rag.guardrails.guarded import HEDGE_PREFIX


class FakeJudge:
    name = "fake-judge@local"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        self.seen.append(answer)
        if question == "boom":
            raise TimeoutError
        return 0.8

    def answer_relevancy(self, question: str, answer: str) -> float:
        return 0.6


def _row(qid: str, provenance: str, question: str = "q", answered: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": qid,
        "query": question,
        "provenance": provenance,
        "answer": HEDGE_PREFIX + "An answer.",
        "contexts": ["ctx"],
    }
    if answered:
        row["groundedness"] = 0.5
    return row


def test_scores_answered_rows_per_half_and_counts_failures() -> None:
    rows = [
        _row("h1", "hand"),
        _row("h2", "hand", question="boom"),
        _row("s1", "synthetic"),
        _row("r1", "hand", answered=False),
    ]
    judge = FakeJudge()

    result = run_tier_c(rows, judge)

    assert result.judge == "fake-judge@local"
    # the failed judgement is excluded from the mean, not scored as zero
    assert result.by_provenance["hand"]["faithfulness"] == pytest.approx(0.8)
    assert result.by_provenance["hand"]["answer_relevancy"] == pytest.approx(0.6)
    assert result.counts["hand"] == {
        "judged": 2,
        "faithfulness_failed": 1,
        "answer_relevancy_failed": 0,
    }
    assert result.by_provenance["synthetic"]["faithfulness"] == pytest.approx(0.8)
    assert "overall" not in result.model_dump()
    # the hedge warning is not part of what the judge sees
    assert all(not answer.startswith("⚠") for answer in judge.seen)


def test_limit_caps_the_number_of_judged_rows() -> None:
    result = run_tier_c([_row("h1", "hand"), _row("h2", "hand")], FakeJudge(), limit=1)
    assert len(result.per_query) == 1


def test_a_half_with_every_judgement_failing_reports_no_mean() -> None:
    result = run_tier_c([_row("h1", "hand", question="boom")], FakeJudge())
    assert "faithfulness" not in result.by_provenance["hand"]
    assert result.counts["hand"]["faithfulness_failed"] == 1


def test_default_judge_is_local_and_free() -> None:
    # NFR-9: zero paid-API calls in the default configuration.
    judge = Settings().judge
    assert judge.base_url.startswith("http://localhost")
    assert judge.model is None  # falls back to the local generator


def test_ragas_judge_explains_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_ragas(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("ragas") or name == "openai":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ragas)
    from rag.eval.ragas_judge import RagasJudge

    with pytest.raises(RuntimeError, match="uv sync --extra judge"):
        RagasJudge(Settings())

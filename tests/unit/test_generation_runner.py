from typing import Any
from unittest.mock import MagicMock

import pytest

from rag.contracts import Answer, Chunk, Citation, Retrieved
from rag.eval.generation_runner import run_tier_b
from rag.eval.golden import GoldenQuery
from rag.eval.metrics.generation import ScoredSentence
from rag.guardrails.guarded import HEDGE_PREFIX


class FakeScorer:
    """Every cited marker with chunk text scores 0.9."""

    def score(
        self, sentences: list[tuple[str, list[str]]], chunk_by_marker: dict[str, str]
    ) -> list[ScoredSentence]:
        return [
            ScoredSentence(
                text=text,
                cited_markers=markers,
                support={m: 0.9 for m in markers if m in chunk_by_marker},
            )
            for text, markers in sentences
        ]


class FakeEmbedder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[Any]:
        self.seen.extend(texts)
        return [[[1.0, 0.0]] for _ in texts]  # identical vectors -> F1 = 1.0


def _answered(text: str) -> Answer:
    chunk = Chunk(
        chunk_id="c1", doc_id="d", text="RRF fuses rankings.", source_path="a.py", language="python"
    )
    return Answer(
        text=text,
        retrieved=[Retrieved(chunk=chunk)],
        citations=[Citation(marker="[1]", chunk_id="c1", source_path="a.py", display_path="a.py")],
    )


def test_scores_the_guarded_answer_hand_computed() -> None:
    queries = [
        GoldenQuery(
            id="h1", query="q", provenance="hand", relevant_files=["a.py"], golden_answer="ref"
        ),
        GoldenQuery(id="h2", query="out of scope", provenance="hand", expect_refusal=True),
    ]
    answerer = MagicMock()
    answerer.answer.side_effect = [
        _answered(HEDGE_PREFIX + "RRF fuses rankings [1]. It needs no normalisation."),
        Answer(text="refused", refused=True),
    ]
    embedder = FakeEmbedder()

    result = run_tier_b(queries, answerer, FakeScorer(), embedder, t_pass=0.5)
    hand = result.by_provenance["hand"]

    # sentences score [0.9, 0] -> groundedness 0.45, recall 1/2, precision 1/1
    assert hand["groundedness"] == pytest.approx(0.45)
    assert hand["citation_recall"] == pytest.approx(0.5)
    assert hand["citation_precision"] == pytest.approx(1.0)
    assert hand["bertscore_f1"] == pytest.approx(1.0)
    assert hand["ungrounded_rate"] == pytest.approx(0.0)
    # one correct answer + one correct refusal
    assert hand["refusal_correctness"] == pytest.approx(1.0)
    assert result.counts["hand"]["scored"] == 1
    assert result.counts["hand"]["correct_refusal"] == 1

    row = result.per_query[0]
    # the hedge warning is not part of the answer being scored
    assert not row["sentences"][0]["text"].startswith("⚠")
    # BERTScore compares claim text, never citation markers
    assert "[1]" not in embedder.seen[0]
    assert row["contexts"] == ["RRF fuses rankings."]


def test_never_pools_and_omits_metrics_without_data() -> None:
    queries = [GoldenQuery(id="s1", query="q", provenance="synthetic", expect_refusal=True)]
    answerer = MagicMock()
    answerer.answer.return_value = Answer(text="an answer", refused=False)

    result = run_tier_b(queries, answerer, FakeScorer(), FakeEmbedder(), t_pass=0.5)

    assert "overall" not in result.model_dump()
    assert result.by_provenance["synthetic"] == {"refusal_correctness": 0.0}
    assert result.counts["synthetic"]["missed_refusal"] == 1
    assert result.by_provenance["hand"] == {}


def test_a_false_refusal_is_counted_and_not_scored() -> None:
    queries = [GoldenQuery(id="h1", query="q", provenance="hand", relevant_files=["a.py"])]
    answerer = MagicMock()
    answerer.answer.return_value = Answer(text="blocked", refused=True)

    result = run_tier_b(queries, answerer, FakeScorer(), FakeEmbedder(), t_pass=0.5)

    assert result.counts["hand"]["false_refusal"] == 1
    assert result.counts["hand"]["scored"] == 0
    assert "groundedness" not in result.by_provenance["hand"]

"""T1 topicality rail — cosine similarity to the corpus centroid (FR-GR1).

ADR-015 assigned out-of-scope refusal to this rail, because the RRF fused score provably
cannot do it. Unlike that score, similarity to the corpus centroid is calibrated.

The rail reuses the retrieval embedder, so it adds no model residency.
"""

from __future__ import annotations

import math

import pytest

from rag.contracts import RailContext
from rag.guardrails.policy import RailPolicy
from rag.guardrails.rails.topicality import TopicalityRail, cosine


class StubEmbedder:
    """Maps known queries to fixed unit vectors, as the real embedder normalises."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vectors[text]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


ON_TOPIC = _unit([1.0, 0.05, 0.0])
BORDERLINE = _unit([1.0, 0.0, 3.0])  # cosine 0.316 to the centroid — inside (0.20, 0.45)
OFF_TOPIC = _unit([0.0, 0.0, 1.0])
CENTROID = _unit([1.0, 0.0, 0.0])


def _rail(query: str, vector: list[float]) -> TopicalityRail:
    return TopicalityRail(
        RailPolicy(action="refuse", on_error="open", t_pass=0.45, t_block=0.20),
        embedder=StubEmbedder({query: vector}),
        centroid_provider=lambda: CENTROID,
    )


def test_cosine_of_identical_vectors_is_one() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_normalises_an_unnormalised_centroid() -> None:
    """The mean of unit vectors is not itself a unit vector."""
    assert cosine([1.0, 0.0], [3.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_a_zero_vector_is_zero_not_a_crash() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_an_on_topic_question_passes() -> None:
    result = _rail("q", ON_TOPIC).check(RailContext(request_id="r", query="q"))
    assert result.verdict == "pass"
    assert result.tier == "T1"


def test_an_off_topic_question_is_refused() -> None:
    result = _rail("q", OFF_TOPIC).check(RailContext(request_id="r", query="q"))
    assert result.verdict == "refuse"


def test_a_borderline_question_hedges_rather_than_refusing() -> None:
    """Refusing on uncertainty is how a guardrail acquires a false-refusal rate."""
    result = _rail("q", BORDERLINE).check(RailContext(request_id="r", query="q"))
    assert result.verdict == "hedge"


def test_the_similarity_is_reported_as_the_score() -> None:
    result = _rail("q", ON_TOPIC).check(RailContext(request_id="r", query="q"))
    assert result.score == pytest.approx(cosine(ON_TOPIC, CENTROID))


def test_the_band_is_reported() -> None:
    assert _rail("q", ON_TOPIC).check(RailContext(request_id="r", query="q")).threshold_band == (
        0.20,
        0.45,
    )


def test_an_empty_corpus_centroid_fails_open() -> None:
    """No centroid means nothing to compare against. Refusing everything would be worse."""
    rail = TopicalityRail(
        RailPolicy(action="refuse", on_error="open", t_pass=0.45, t_block=0.20),
        embedder=StubEmbedder({"q": ON_TOPIC}),
        centroid_provider=lambda: None,
    )
    result = rail.check(RailContext(request_id="r", query="q"))
    assert result.verdict == "skipped"
    assert result.evidence["reason"] == "no_centroid"


def test_latency_is_recorded() -> None:
    assert _rail("q", ON_TOPIC).check(RailContext(request_id="r", query="q")).latency_ms >= 0.0

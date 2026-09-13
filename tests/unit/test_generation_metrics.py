import pytest

from rag.eval.metrics.generation import (
    ScoredSentence,
    bertscore_f1,
    citation_precision,
    citation_recall,
    groundedness,
    refusal_correctness,
    refusal_outcome,
)


def _two_sentences() -> list[ScoredSentence]:
    return [
        ScoredSentence(
            text="A [1][2].", cited_markers=["[1]", "[2]"], support={"[1]": 0.9, "[2]": 0.4}
        ),
        ScoredSentence(text="B.", cited_markers=[], support={}),
    ]


def test_groundedness_takes_max_per_sentence_and_zero_for_uncited() -> None:
    # (max(0.9, 0.4) + 0) / 2 = 0.45
    assert groundedness(_two_sentences()) == pytest.approx(0.45)


def test_groundedness_of_an_empty_answer_is_zero() -> None:
    assert groundedness([]) == 0.0


def test_citation_precision_counts_pairs_at_or_above_t_pass() -> None:
    # pairs: 0.9 (>= 0.5), 0.4 (< 0.5) -> 1/2
    assert citation_precision(_two_sentences(), t_pass=0.5) == pytest.approx(0.5)
    # the boundary is inclusive: 0.9 and 0.4 are both >= 0.4 -> 2/2
    assert citation_precision(_two_sentences(), t_pass=0.4) == pytest.approx(1.0)


def test_citation_precision_is_undefined_without_citations() -> None:
    uncited = [ScoredSentence(text="B.", cited_markers=[], support={})]
    assert citation_precision(uncited, t_pass=0.5) is None


def test_citation_recall_counts_sentences_with_a_supporting_citation() -> None:
    # sentence 1 has 0.9 >= 0.5; sentence 2 has nothing -> 1/2
    assert citation_recall(_two_sentences(), t_pass=0.5) == pytest.approx(0.5)
    assert citation_recall([], t_pass=0.5) == 0.0


def test_refusal_outcomes_cover_all_four_cells() -> None:
    assert refusal_outcome(expect_refusal=False, refused=False) == "correct_answer"
    assert refusal_outcome(expect_refusal=True, refused=True) == "correct_refusal"
    assert refusal_outcome(expect_refusal=False, refused=True) == "false_refusal"
    assert refusal_outcome(expect_refusal=True, refused=False) == "missed_refusal"


def test_refusal_correctness_is_the_fraction_of_correct_cells() -> None:
    outcomes = ["correct_answer", "missed_refusal", "correct_refusal", "false_refusal"]
    assert refusal_correctness(outcomes) == 0.5  # type: ignore[arg-type]
    assert refusal_correctness([]) == 0.0


def test_bertscore_identical_token_sets_score_one() -> None:
    assert bertscore_f1([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)


def test_bertscore_partial_match_hand_computed() -> None:
    # cand [[1,0],[0,1]], ref [[1,0]]: sim = [[1],[0]]
    # precision = mean(1, 0) = 0.5; recall = max(1, 0) = 1.0; F1 = 2*0.5*1/1.5 = 2/3
    assert bertscore_f1([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]]) == pytest.approx(2 / 3)


def test_bertscore_is_scale_invariant() -> None:
    assert bertscore_f1([[3.0, 0.0]], [[1.0, 0.0]]) == pytest.approx(1.0)


def test_bertscore_degenerate_inputs_score_zero() -> None:
    assert bertscore_f1([], [[1.0, 0.0]]) == 0.0
    assert bertscore_f1([[1.0, 0.0]], [[0.0, 1.0]]) == 0.0

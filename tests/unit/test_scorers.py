from unittest.mock import MagicMock

import pytest

from rag.config import Settings
from rag.eval.metrics.generation import bertscore_f1
from rag.eval.scorers import HHEMSupportScorer, TokenEmbedder, split_answer


def test_split_answer_attaches_markers_to_their_sentence() -> None:
    assert split_answer("RRF fuses rankings [1][2]. It needs no normalisation.") == [
        ("RRF fuses rankings [1][2].", ["[1]", "[2]"]),
        ("It needs no normalisation.", []),
    ]


def test_split_answer_deduplicates_repeated_markers() -> None:
    assert split_answer("A [1] and again [1].") == [("A [1] and again [1].", ["[1]"])]


def test_hhem_scorer_pairs_each_cited_chunk_separately_and_strips_markers() -> None:
    model = MagicMock()
    model.predict.return_value = [0.9, 0.2]
    scorer = HHEMSupportScorer(Settings(), model=model)

    scored = scorer.score(
        [("RRF fuses [1][2].", ["[1]", "[2]"]), ("Uncited.", []), ("Stale [9].", ["[9]"])],
        {"[1]": "chunk one", "[2]": "chunk two"},
    )

    # (premise, hypothesis) per cited chunk; never concatenated; markers stripped.
    assert model.predict.call_args.args[0] == [
        ("chunk one", "RRF fuses."),
        ("chunk two", "RRF fuses."),
    ]
    assert scored[0].support == {"[1]": pytest.approx(0.9), "[2]": pytest.approx(0.2)}
    assert scored[1].support == {}
    assert scored[2].support == {}  # a marker with no chunk text is not scored


def test_hhem_scorer_skips_the_model_when_nothing_is_cited() -> None:
    model = MagicMock()
    HHEMSupportScorer(Settings(), model=model).score([("Uncited.", [])], {})
    model.predict.assert_not_called()


def test_token_embedder_uses_the_configured_model_and_layer() -> None:
    settings = Settings()
    embedder = TokenEmbedder(settings, model=MagicMock(), tokenizer=MagicMock())
    assert embedder.model_name == "distilbert/distilbert-base-uncased"
    assert embedder.layer == 5


@pytest.mark.slow
def test_real_hhem_prefers_a_supported_claim() -> None:
    scorer = HHEMSupportScorer(Settings())
    premise = {"[1]": "Reciprocal Rank Fusion sums 1/(k + rank) across retrievers."}
    supported, contradicted = scorer.score(
        [
            ("Fusion sums 1/(k + rank) across retrievers [1].", ["[1]"]),
            ("Fusion averages raw cosine scores [1].", ["[1]"]),
        ],
        premise,
    )
    assert supported.support["[1]"] > contradicted.support["[1]"]


@pytest.mark.slow
def test_real_token_embedder_scores_paraphrase_above_unrelated() -> None:
    embed = TokenEmbedder(Settings()).embed
    ref, para, other = embed(
        [
            "The retriever fuses dense and sparse results.",
            "Dense and sparse results are fused by the retriever.",
            "Bake the bread at two hundred degrees.",
        ]
    )
    assert bertscore_f1(para, ref) > bertscore_f1(other, ref)

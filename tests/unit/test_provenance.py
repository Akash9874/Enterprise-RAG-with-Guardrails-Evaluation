from pathlib import Path

from rag.config import Settings
from rag.eval.golden import GoldenQuery
from rag.eval.provenance import Provenance, collect_provenance, summarize_commits


def _valid(**overrides: object) -> Provenance:
    base: dict[str, object] = {
        "corpus_commit": "abc1234",
        "config_hash": "9f8e7d6c",
        "policy_hash": "11223344",
        "golden_set_hash": "aabbccddeeff",
        "golden_set": {"hand": 1, "synthetic": 0},
        "models": {"embedder": "BAAI/bge-small-en-v1.5"},
    }
    base.update(overrides)
    return Provenance.model_validate(base)


def test_complete_provenance_has_no_problems() -> None:
    assert _valid().problems() == []


def test_unknown_commit_is_a_problem() -> None:
    assert "corpus_commit is unknown" in _valid(corpus_commit="unknown").problems()


def test_an_index_built_from_several_commits_is_a_problem() -> None:
    problems = _valid(corpus_commit="mixed:abc1234,def5678").problems()
    assert any("more than one commit" in p for p in problems)


def test_tier_c_without_a_judge_is_a_problem() -> None:
    # A Tier C number without its judge is the misleading number eval/CLAUDE.md forbids.
    assert _valid(tier_c_enabled=True, judge=None).problems() == ["tier C enabled without a judge"]


def test_summarize_commits_single_mixed_and_empty() -> None:
    assert summarize_commits({"abc1234"}) == "abc1234"
    # Sorted, so the same index always yields the same provenance string.
    assert summarize_commits({"def5678", "abc1234"}) == "mixed:abc1234,def5678"
    assert summarize_commits({"abc1234", "unknown"}) == "mixed:abc1234,unknown"
    assert summarize_commits(set()) == "unknown"


def test_collect_records_the_commit_it_is_given_models_counts_and_hashes(tmp_path: Path) -> None:
    golden = tmp_path / "g.yaml"
    golden.write_text("- id: q1\n", encoding="utf-8")
    queries = [
        GoldenQuery(id="h", query="?", provenance="hand", relevant_files=["a"]),
        GoldenQuery(id="s", query="?", provenance="synthetic", relevant_chunk_ids=["b"]),
    ]

    prov = collect_provenance(Settings(), queries, golden, corpus_commit="07b032b")

    # The indexed tree's commit, never the process's checkout (ADR-026).
    assert prov.corpus_commit == "07b032b"
    assert prov.golden_set == {"hand": 1, "synthetic": 1}
    assert prov.models["embedder"] == "BAAI/bge-small-en-v1.5"
    assert prov.models["groundedness"] == "vectara/hallucination_evaluation_model"
    assert "registry_max_resident" not in prov.models
    assert len(prov.golden_set_hash) == 12
    assert len(prov.policy_hash) == 8
    assert prov.judge is None and prov.tier_c_enabled is False
    assert prov.problems() == []

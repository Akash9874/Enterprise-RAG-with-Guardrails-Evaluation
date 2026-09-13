from pathlib import Path

import pytest

from rag.config import Settings, load_settings, project_path


def test_project_path_anchors_relative_paths_to_the_repo_root() -> None:
    resolved = project_path("eval/golden/golden.yaml")
    assert resolved.is_absolute()
    assert resolved.parts[-3:] == ("eval", "golden", "golden.yaml")
    assert project_path(str(resolved)) == resolved


def test_golden_path_is_configured() -> None:
    assert Settings().eval.golden_path == "eval/golden/golden.yaml"


def test_ingest_source_can_be_overridden_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # docker-compose points the "self" source at the read-only /corpus mount this way.
    monkeypatch.setenv("RAG_INGEST__SOURCES__SELF", "/corpus")
    assert load_settings().ingest.sources["self"] == "/corpus"


def test_remote_code_model_is_pinned_to_a_full_commit_sha() -> None:
    # HHEM loads with trust_remote_code. A branch name like "main" would not pin anything.
    import re

    revision = Settings().models.groundedness_revision
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert load_settings().models.groundedness_revision == revision  # YAML agrees


def test_ingest_sources_are_keys_not_paths_by_default() -> None:
    assert Settings().ingest.sources == {"self": "."}


def test_defaults_apply_when_no_yaml_present(tmp_path: Path) -> None:
    settings = load_settings(config_path=tmp_path / "missing.yaml")
    assert settings.retrieval.k_final == 5
    assert settings.models.embedder == "BAAI/bge-small-en-v1.5"


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("retrieval:\n  k_final: 9\n", encoding="utf-8")
    settings = load_settings(config_path=path)
    assert settings.retrieval.k_final == 9


def test_env_var_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("retrieval:\n  k_final: 9\n", encoding="utf-8")
    monkeypatch.setenv("RAG_RETRIEVAL__K_FINAL", "3")
    settings = load_settings(config_path=path)
    assert settings.retrieval.k_final == 3


def test_generator_model_is_a_3b_class_model() -> None:
    """Guards the hardware constraint in PRD section 4: 7B+ is not viable on this CPU."""
    settings = Settings()
    assert "3b" in settings.models.generator.lower()


def test_settings_hash_is_stable_and_content_sensitive() -> None:
    a = Settings()
    b = Settings()
    assert a.config_hash() == b.config_hash()
    c = Settings.model_validate({"retrieval": {"k_final": 7}})
    assert c.config_hash() != a.config_hash()


def test_default_config_path_is_absolute_so_cwd_cannot_break_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative default path silently falls back to field defaults when the process
    starts from anywhere but the project root -- config vanishes with no error."""
    monkeypatch.chdir(tmp_path)
    from rag.config import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH.is_absolute()
    assert DEFAULT_CONFIG_PATH.exists()


def test_settings_load_correctly_from_an_unrelated_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.qdrant.vector_size == 384
    assert settings.ollama.temperature == 0.0

from pathlib import Path

import pytest

from rag.config import Settings, load_settings


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

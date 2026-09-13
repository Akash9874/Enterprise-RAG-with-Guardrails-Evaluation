"""Layered configuration: field defaults -> config/settings.yaml -> RAG_* env vars."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import structlog
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

log = structlog.get_logger(__name__)


def _project_root() -> Path:
    """Locate the project root by walking up to the directory holding pyproject.toml.

    Anchoring on the package file rather than the process working directory: a relative
    default path resolves to nothing when the process starts anywhere else, and the YAML
    is then skipped in silence, leaving every setting at its field default.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


DEFAULT_CONFIG_PATH = _project_root() / "config" / "settings.yaml"


def project_path(value: str) -> Path:
    """Resolve a configured path against the project root unless it is already absolute."""
    path = Path(value)
    return path if path.is_absolute() else _project_root() / path


class ModelSettings(BaseModel):
    embedder: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator: str = "qwen2.5:3b-instruct-q4_K_M"
    groundedness: str = "vectara/hallucination_evaluation_model"
    injection: str = "protectai/deberta-v3-base-prompt-injection-v2"
    # Tier B BERTScore. 268 MB; bert-score's deberta-xlarge-mnli is 3,036 MB (ADR-024).
    bertscore: str = "distilbert/distilbert-base-uncased"
    registry_max_resident: int = 4


class RetrievalSettings(BaseModel):
    k_dense: int = 20
    k_sparse: int = 20
    k_fuse: int = 30
    k_final: int = 5
    rrf_k: int = 60
    context_budget_tokens: int = 2400
    relevance_floor: float = 0.0
    # Measured negative lift and ~2.3 s/query cost on the real corpus — see ADR-003.
    rerank_enabled: bool = False


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "rag_corpus"
    vector_size: int = 384


class OllamaSettings(BaseModel):
    host: str = "http://localhost:11434"
    timeout_s: int = 300
    temperature: float = 0.0


class EvalSettings(BaseModel):
    golden_path: str = "eval/golden/golden.yaml"
    # bert-score's default num_layers for distilbert-base-uncased.
    bertscore_layer: int = 5
    # Written only by `rag eval promote-baseline` (FR-E8).
    baseline_path: str = "eval/baselines/baseline.json"
    reports_dir: str = "eval/reports"
    # FR-E7. Absolute drop on a 0-1 scale, checked per provenance half, never pooled (ADR-025).
    gate_max_drop: dict[str, float] = Field(
        default_factory=lambda: {
            "tier_a.by_provenance.hand.recall@5": 0.02,
            "tier_a.by_provenance.synthetic.recall@5": 0.02,
            "tier_b.by_provenance.hand.groundedness": 0.03,
            "tier_b.by_provenance.synthetic.groundedness": 0.03,
        }
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    models: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Precedence, highest first: init kwargs > env vars > YAML > field defaults.

        Passing YAML as init kwargs would invert this and let the file silently beat the
        environment, which is the opposite of what deployment expects.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def config_hash(self) -> str:
        """Short digest of the effective config. Stamped into every eval report (FR-E4)."""
        payload = self.model_dump_json()
        return hashlib.blake2b(payload.encode(), digest_size=4).hexdigest()


def load_settings(config_path: Path | None = None) -> Settings:
    """Build settings from a YAML file, env vars, and defaults.

    The YAML path is bound through a subclass because `settings_customise_sources` is a
    classmethod with no access to per-call arguments. `Settings()` on its own reads env
    vars and defaults only.
    """
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        # Never fall back in silence -- a missing config file looks identical to one
        # whose every value happens to match the defaults.
        log.warning("config_file_missing", path=str(path), using="field defaults")

    class _Configured(Settings):
        model_config = SettingsConfigDict(
            env_prefix="RAG_",
            env_nested_delimiter="__",
            extra="ignore",
            yaml_file=path,
        )

    return _Configured()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()

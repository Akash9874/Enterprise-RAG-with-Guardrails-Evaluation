"""Report provenance (FR-E4). A report without a valid block is never written.

The corpus commit is supplied by the caller from the index being scored — the stamps ingest
wrote onto every chunk — never guessed from the eval process's own checkout (ADR-026).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from rag.config import Settings
from rag.eval.golden import GoldenQuery, golden_set_hash
from rag.gitinfo import UNKNOWN
from rag.guardrails.policy import DEFAULT_POLICY_PATH

MIXED_PREFIX = "mixed:"


class Provenance(BaseModel):
    corpus_commit: str
    config_hash: str
    policy_hash: str
    golden_set_hash: str
    golden_set: dict[str, int]
    models: dict[str, str]
    judge: str | None = None
    tier_c_enabled: bool = False

    def problems(self) -> list[str]:
        issues: list[str] = []
        if self.corpus_commit in {"", UNKNOWN}:
            issues.append("corpus_commit is unknown")
        elif self.corpus_commit.startswith(MIXED_PREFIX):
            issues.append(
                f"the index spans more than one commit ({self.corpus_commit}); "
                "re-ingest with --recreate"
            )
        for name in ("config_hash", "policy_hash", "golden_set_hash"):
            if not getattr(self, name):
                issues.append(f"{name} is empty")
        if not self.models:
            issues.append("models is empty")
        if self.tier_c_enabled and not self.judge:
            issues.append("tier C enabled without a judge")
        return issues


def summarize_commits(commits: set[str]) -> str:
    """One commit as-is; several as a sorted `mixed:` list; none as unknown."""
    if not commits:
        return UNKNOWN
    if len(commits) == 1:
        return next(iter(commits))
    return MIXED_PREFIX + ",".join(sorted(commits))


def policy_hash(path: Path | None = None) -> str:
    resolved = path if path is not None else DEFAULT_POLICY_PATH
    text = resolved.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.blake2b(text.encode(), digest_size=4).hexdigest()


def collect_provenance(
    settings: Settings,
    queries: list[GoldenQuery],
    golden_path: Path,
    corpus_commit: str,
    judge: str | None = None,
) -> Provenance:
    models = {k: v for k, v in settings.models.model_dump().items() if isinstance(v, str)}
    return Provenance(
        corpus_commit=corpus_commit,
        config_hash=settings.config_hash(),
        policy_hash=policy_hash(),
        golden_set_hash=golden_set_hash(golden_path),
        golden_set={
            "hand": sum(1 for q in queries if q.provenance == "hand"),
            "synthetic": sum(1 for q in queries if q.provenance == "synthetic"),
        },
        models=models,
        judge=judge,
        tier_c_enabled=judge is not None,
    )

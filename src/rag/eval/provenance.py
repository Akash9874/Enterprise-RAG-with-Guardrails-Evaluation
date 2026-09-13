"""Report provenance (FR-E4). A report without a valid block is never written."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel

from rag.config import Settings
from rag.eval.golden import GoldenQuery, golden_set_hash
from rag.guardrails.policy import DEFAULT_POLICY_PATH


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
        if self.corpus_commit in {"", "unknown"}:
            issues.append("corpus_commit is unknown")
        for name in ("config_hash", "policy_hash", "golden_set_hash"):
            if not getattr(self, name):
                issues.append(f"{name} is empty")
        if not self.models:
            issues.append("models is empty")
        if self.tier_c_enabled and not self.judge:
            issues.append("tier C enabled without a judge")
        return issues


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()


def corpus_commit(root: Path | None = None) -> str:
    """Short SHA, suffixed `-dirty` when tracked files have uncommitted changes.

    A dirty tree's numbers correspond to no commit, and the report must say so.
    """
    where = root if root is not None else Path.cwd()
    try:
        sha = _git(where, "rev-parse", "--short", "HEAD")
        dirty = bool(_git(where, "status", "--porcelain", "--untracked-files=no"))
    except Exception:  # noqa: BLE001 - provenance collection must never raise
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def policy_hash(path: Path | None = None) -> str:
    resolved = path if path is not None else DEFAULT_POLICY_PATH
    text = resolved.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.blake2b(text.encode(), digest_size=4).hexdigest()


def collect_provenance(
    settings: Settings,
    queries: list[GoldenQuery],
    golden_path: Path,
    judge: str | None = None,
) -> Provenance:
    models = {k: v for k, v in settings.models.model_dump().items() if isinstance(v, str)}
    return Provenance(
        corpus_commit=corpus_commit(),
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

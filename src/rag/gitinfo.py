"""Git facts about a directory — the commit of a tree, asked of that tree.

A leaf module with no project imports, so both `ingest` (which stamps chunks) and `eval`
(which reports provenance) can use it without either importing the other (invariant 1).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

UNKNOWN = "unknown"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()


def git_commit(root: Path) -> str:
    """Short SHA of the tree at `root`, suffixed `-dirty` when tracked files are modified.

    Asks the directory being described, never the process's working directory: the
    defect this replaces reported the eval process's checkout even when the index had been
    built from a different tree (ADR-026). Untracked files do not count as dirty.
    """
    try:
        sha = _git(root, "rev-parse", "--short", "HEAD")
        dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    except Exception:  # noqa: BLE001 - provenance collection must never raise
        return UNKNOWN
    return f"{sha}-dirty" if dirty else sha

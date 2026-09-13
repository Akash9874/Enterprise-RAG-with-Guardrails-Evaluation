"""Walk a repository and yield decoded text files, honouring .gitignore."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec
import structlog

log = structlog.get_logger(__name__)

EXTENSION_LANGUAGE = {
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".js": "javascript",
    ".ts": "typescript",
}

ALWAYS_SKIP = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "qdrant_storage",
}

# Evaluation fixtures describe the corpus; they are not part of it. Indexing them let a
# golden query retrieve its own question text (ADR-023).
ALWAYS_SKIP_PREFIXES = ("eval/",)


@dataclass(frozen=True)
class LoadedFile:
    path: str  # POSIX, relative to root
    text: str
    language: str


def _gitignore_spec(root: Path) -> pathspec.PathSpec[pathspec.Pattern] | None:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def load_repository(root: Path) -> Iterator[LoadedFile]:
    spec = _gitignore_spec(root)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(root)
        if any(part in ALWAYS_SKIP or part.startswith(".") for part in relative.parts[:-1]):
            continue
        if relative.name.startswith(".") and relative.suffix not in EXTENSION_LANGUAGE:
            continue

        language = EXTENSION_LANGUAGE.get(path.suffix.lower())
        if language is None:
            log.debug("skipped_extension", path=str(relative))
            continue

        posix = relative.as_posix()
        if posix.startswith(ALWAYS_SKIP_PREFIXES):
            continue
        if spec is not None and spec.match_file(posix):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.debug("skipped_undecodable", path=posix)
            continue

        yield LoadedFile(path=posix, text=text, language=language)

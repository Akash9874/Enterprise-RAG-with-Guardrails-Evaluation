import subprocess
from pathlib import Path

from rag.gitinfo import UNKNOWN, git_commit


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    # `git -C` refuses a directory that does not exist yet (exit 128).
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_directory_outside_any_repository_is_unknown(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == UNKNOWN


def test_a_clean_repository_reports_its_short_sha(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert git_commit(root) == _git(root, "rev-parse", "--short", "HEAD")


def test_modified_tracked_files_mark_the_commit_dirty(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert git_commit(root).endswith("-dirty")


def test_untracked_files_do_not_mark_it_dirty(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "scratch.txt").write_text("notes\n", encoding="utf-8")
    assert not git_commit(root).endswith("-dirty")


def test_the_commit_belongs_to_the_directory_asked_about(tmp_path: Path) -> None:
    # The defect this module exists to fix: provenance used the process's working
    # checkout, not the tree that was actually indexed (ADR-026).
    first = _repo(tmp_path / "one")
    second = _repo(tmp_path / "two")
    (second / "b.py").write_text("y = 1\n", encoding="utf-8")
    _git(second, "add", "b.py")
    _git(second, "commit", "-q", "-m", "second")
    assert git_commit(first) != git_commit(second)

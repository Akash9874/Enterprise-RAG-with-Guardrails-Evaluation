from pathlib import Path

from rag.ingest.loaders import load_repository


def _write(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_loads_python_and_markdown_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "b.md", "# Title")
    loaded = {f.path: f for f in load_repository(tmp_path)}
    assert set(loaded) == {"a.py", "b.md"}
    assert loaded["a.py"].language == "python"
    assert loaded["b.md"].language == "markdown"


def test_skips_unknown_extensions(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "image.png", "not really a png")
    assert [f.path for f in load_repository(tmp_path)] == ["a.py"]


def test_honours_gitignore(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "secret.py\nbuild/\n")
    _write(tmp_path, "keep.py", "x = 1")
    _write(tmp_path, "secret.py", "password = 'hunter2'")
    _write(tmp_path, "build/out.py", "generated = True")
    assert [f.path for f in load_repository(tmp_path)] == ["keep.py"]


def test_always_skips_dot_directories(tmp_path: Path) -> None:
    _write(tmp_path, "keep.py", "x = 1")
    _write(tmp_path, ".venv/lib/mod.py", "y = 2")
    _write(tmp_path, ".git/config", "[core]")
    assert [f.path for f in load_repository(tmp_path)] == ["keep.py"]


def test_paths_are_posix_relative_to_root(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/mod.py", "x = 1")
    assert [f.path for f in load_repository(tmp_path)] == ["src/pkg/mod.py"]


def test_eval_fixtures_are_never_indexed(tmp_path: Path) -> None:
    # Golden answers and adversarial payloads must never be retrievable context.
    _write(tmp_path, "eval/golden/golden.yaml", "- id: q1\n")
    _write(tmp_path, "eval/adversarial/suite.yaml", "- id: adv\n")
    _write(tmp_path, "src/a.py", "x = 1\n")
    assert [f.path for f in load_repository(tmp_path)] == ["src/a.py"]


def test_skips_files_that_are_not_valid_utf8(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", "x = 1")
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00binary")
    assert [f.path for f in load_repository(tmp_path)] == ["good.py"]

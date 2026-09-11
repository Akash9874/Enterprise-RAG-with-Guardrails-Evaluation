from pathlib import Path

from rag.ingest.pipeline import build_chunks


def _write(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_dispatches_code_and_markdown_to_the_right_chunkers(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "def f():\n    return 1\n")
    _write(tmp_path, "doc.md", "# Title\n\nBody text.\n")
    chunks, _ = build_chunks(tmp_path)

    by_lang = {c.language for c in chunks}
    assert by_lang == {"python", "markdown"}
    assert any(c.symbol_path == "f" for c in chunks)
    assert any(c.header_path == "Title" for c in chunks)


def test_reports_accurate_stats(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    _write(tmp_path, "b.md", "# T\n\nBody.\n")
    chunks, stats = build_chunks(tmp_path)
    assert stats.files == 2
    assert stats.chunks == len(chunks)
    assert stats.duration_s >= 0.0


def test_is_idempotent_across_runs(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    first, _ = build_chunks(tmp_path)
    second, _ = build_chunks(tmp_path)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_deduplicates_identical_chunk_ids(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    chunks, _ = build_chunks(tmp_path)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_empty_repository_yields_no_chunks(tmp_path: Path) -> None:
    chunks, stats = build_chunks(tmp_path)
    assert chunks == []
    assert stats.files == 0

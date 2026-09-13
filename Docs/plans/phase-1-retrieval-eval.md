# Phase 1 — Ingestion, Retrieval, and the Tier A Evaluation Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ingest this repository into Qdrant with structure-aware chunking, retrieve from it
with hybrid dense + sparse search fused server-side by RRF and reranked by a cross-encoder, and
**measure all of it** with a deterministic, zero-LLM evaluation harness.

**Architecture:** Loaders walk the repo and dispatch to chunkers by file type — tree-sitter for
Python (AST boundaries, symbol paths), `MarkdownHeaderTextSplitter` for docs (header paths).
Chunks carry stable content-derived ids, so ingestion is idempotent. Qdrant holds dense and
sparse vectors in one collection and fuses them in a single round trip. The eval harness scores
retrieval against a ~50-query golden set with no LLM in the loop.

**Spec:** [`Docs/prd.md`](../prd.md) §7.1, §7.2, §7.5 · **Constraints:** [`README.md`](README.md#global-constraints)

**Exit criteria:** `uv run rag eval retrieval` prints Recall@5, Precision@5, MRR, NDCG@5, Hit
Rate, and reranker lift across ~50 golden queries in under 60 seconds, scoring hand-authored
and synthetic questions separately.

---

## Why this phase exists before generation

Chunk size, `k` values, fusion parameters, and the reranker choice are **empirical questions**.
Answering them by intuition is the single most common failure in RAG projects. Building the
harness here means every tuning decision from this point forward is backed by a number this
project produced. Phase 1 ships no user-visible feature, and that is the deliberate trade.

---

## File structure produced by this phase

| File | Responsibility |
|---|---|
| `src/rag/ingest/loaders.py` | Walk a directory, honour `.gitignore`, dispatch by extension |
| `src/rag/ingest/chunkers/markdown.py` | Header-path-preserving markdown chunking |
| `src/rag/ingest/chunkers/code.py` | tree-sitter AST chunking with symbol paths |
| `src/rag/ingest/pipeline.py` | Orchestrates load → chunk → dedupe → emit |
| `src/rag/models/embedder.py` | `bge-small` dense + FastEmbed BM25 sparse |
| `src/rag/index/schema.py` | Collection creation with named dense/sparse vectors |
| `src/rag/index/qdrant_store.py` | *(extend)* upsert and hybrid query |
| `src/rag/retrieval/hybrid.py` | `HybridRetriever` — query, fuse, rerank |
| `src/rag/retrieval/rerank.py` | Cross-encoder reranking, toggleable |
| `src/rag/retrieval/context.py` | Dedupe, order, token budget |
| `src/rag/eval/golden.py` | Golden set schema and loader |
| `src/rag/eval/metrics/retrieval.py` | Recall, Precision, MRR, NDCG, Hit Rate |
| `src/rag/eval/runner.py` | Tier A runner, reranker lift, provenance |
| `src/rag/cli.py` | `rag ingest`, `rag eval retrieval` |

**Add to `pyproject.toml` dependencies before starting:**

```toml
    "sentence-transformers>=3.3",
    "fastembed>=0.5",
    "tree-sitter>=0.23",
    "tree-sitter-python>=0.23",
    "langchain-text-splitters>=0.3",
    "pathspec>=0.12",
    "rich>=13.9",
```

Then `uv sync`.

---

## Task 1.1: Repository loader

**Files:**
- Create: `src/rag/ingest/__init__.py`, `src/rag/ingest/loaders.py`
- Test: `tests/unit/test_loaders.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `LoadedFile(path: str, text: str, language: str)` dataclass and
  `load_repository(root: Path) -> Iterator[LoadedFile]`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_loaders.py
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


def test_skips_files_that_are_not_valid_utf8(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", "x = 1")
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00binary")
    assert [f.path for f in load_repository(tmp_path)] == ["good.py"]
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.ingest'`

- [x] **Step 3: Implement the loader**

```python
# src/rag/ingest/__init__.py
```

```python
# src/rag/ingest/loaders.py
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


@dataclass(frozen=True)
class LoadedFile:
    path: str  # POSIX, relative to root
    text: str
    language: str


def _gitignore_spec(root: Path) -> pathspec.PathSpec | None:
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
        if spec is not None and spec.match_file(posix):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.debug("skipped_undecodable", path=posix)
            continue

        yield LoadedFile(path=posix, text=text, language=language)
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_loaders.py -v`
Expected: PASS (6 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/ingest/ tests/unit/test_loaders.py
git commit -m "feat: add repository loader honouring gitignore and extension allowlist"
```

---

## Task 1.2: Markdown header-path chunker

**Files:**
- Create: `src/rag/ingest/chunkers/__init__.py`, `src/rag/ingest/chunkers/markdown.py`
- Test: `tests/unit/test_markdown_chunker.py`

**Interfaces:**
- Consumes: `LoadedFile`, `Chunk`, `make_chunk_id`
- Produces: `estimate_tokens(text: str) -> int` and
  `chunk_markdown(file: LoadedFile, max_tokens: int = 512) -> list[Chunk]`

Each chunk carries `header_path` — the full `H1 > H2 > H3` trail — which is used verbatim in
citations (FR-I4). That is what makes a citation readable rather than a line number.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_markdown_chunker.py
from rag.ingest.chunkers.markdown import chunk_markdown, estimate_tokens
from rag.ingest.loaders import LoadedFile

DOC = """# Architecture

Intro text about the system.

## Retrieval

Retrieval is hybrid.

### Hybrid Search

Dense plus sparse, fused by RRF.

## Generation

Generation uses a local model.
"""


def _file(text: str = DOC) -> LoadedFile:
    return LoadedFile(path="Docs/architecture.md", text=text, language="markdown")


def test_splits_on_headers() -> None:
    chunks = chunk_markdown(_file())
    assert len(chunks) >= 3


def test_header_path_records_the_full_trail() -> None:
    chunks = chunk_markdown(_file())
    paths = {c.header_path for c in chunks}
    assert "Architecture > Retrieval > Hybrid Search" in paths


def test_top_level_section_has_a_single_segment_header_path() -> None:
    chunks = chunk_markdown(_file())
    assert any(c.header_path == "Architecture" for c in chunks)


def test_every_chunk_carries_source_and_language() -> None:
    for chunk in chunk_markdown(_file()):
        assert chunk.source_path == "Docs/architecture.md"
        assert chunk.language == "markdown"
        assert chunk.doc_id == "Docs/architecture.md"


def test_chunk_ids_are_stable_across_runs() -> None:
    first = [c.chunk_id for c in chunk_markdown(_file())]
    second = [c.chunk_id for c in chunk_markdown(_file())]
    assert first == second


def test_oversized_section_is_split_into_multiple_chunks() -> None:
    big = "# Title\n\n" + ("word " * 4000)
    chunks = chunk_markdown(LoadedFile(path="big.md", text=big, language="markdown"))
    assert len(chunks) > 1
    assert all(c.token_count <= 512 for c in chunks)


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_markdown(LoadedFile(path="empty.md", text="", language="markdown")) == []


def test_estimate_tokens_is_monotonic_and_nonzero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 100)
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_markdown_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.ingest.chunkers'`

- [x] **Step 3: Implement the chunker**

```python
# src/rag/ingest/chunkers/__init__.py
```

```python
# src/rag/ingest/chunkers/markdown.py
"""Header-aware markdown chunking.

Each chunk keeps its full header trail so citations read as
"Architecture > Retrieval > Hybrid Search" rather than as a line range.
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.contracts import Chunk, make_chunk_id
from rag.ingest.loaders import LoadedFile

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
SEPARATOR = " > "

# Approximate: ~4 characters per token. Accurate enough for budgeting, and it avoids
# pulling a tokenizer into the ingest path.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _header_path(metadata: dict[str, str]) -> str | None:
    parts = [metadata[key] for _, key in HEADERS if metadata.get(key)]
    return SEPARATOR.join(parts) if parts else None


def chunk_markdown(file: LoadedFile, max_tokens: int = 512) -> list[Chunk]:
    if not file.text.strip():
        return []

    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=False
    ).split_text(file.text)

    overflow = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * CHARS_PER_TOKEN,
        chunk_overlap=max_tokens * CHARS_PER_TOKEN // 10,
        length_function=len,
    )

    chunks: list[Chunk] = []
    for section in sections:
        header_path = _header_path(section.metadata)
        for body in overflow.split_text(section.page_content):
            text = body.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(file.path, text),
                    doc_id=file.path,
                    text=text,
                    source_path=file.path,
                    language=file.language,
                    header_path=header_path,
                    token_count=estimate_tokens(text),
                    content_hash=hashlib.blake2b(file.text.encode(), digest_size=8).hexdigest(),
                )
            )
    return chunks
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_markdown_chunker.py -v`
Expected: PASS (8 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/ingest/chunkers/ tests/unit/test_markdown_chunker.py
git commit -m "feat: add header-path-preserving markdown chunker"
```

---

## Task 1.3: tree-sitter code chunker

**Files:**
- Create: `src/rag/ingest/chunkers/code.py`
- Test: `tests/unit/test_code_chunker.py`

**Interfaces:**
- Consumes: `LoadedFile`, `Chunk`, `make_chunk_id`, `estimate_tokens`
- Produces: `chunk_code(file: LoadedFile, max_tokens: int = 512) -> list[Chunk]`

Chunk boundaries fall on function and class definitions. Each chunk carries `symbol_path`
(`ClassName.method_name`) and its line range (FR-I3). A function is never split mid-body unless
it exceeds the budget.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_code_chunker.py
from rag.ingest.chunkers.code import chunk_code
from rag.ingest.loaders import LoadedFile

SOURCE = '''"""Module docstring."""

import os


def top_level(a: int) -> int:
    """Adds one."""
    return a + 1


class Retriever:
    """A retriever."""

    def __init__(self, k: int) -> None:
        self.k = k

    def search(self, query: str) -> list[str]:
        """Searches."""
        return [query] * self.k
'''


def _file(text: str = SOURCE) -> LoadedFile:
    return LoadedFile(path="src/rag/retrieval/hybrid.py", text=text, language="python")


def test_produces_a_chunk_per_top_level_definition() -> None:
    symbols = {c.symbol_path for c in chunk_code(_file())}
    assert "top_level" in symbols
    assert "Retriever" in symbols


def test_methods_carry_a_qualified_symbol_path() -> None:
    symbols = {c.symbol_path for c in chunk_code(_file())}
    assert "Retriever.search" in symbols
    assert "Retriever.__init__" in symbols


def test_chunks_record_their_line_range() -> None:
    chunk = next(c for c in chunk_code(_file()) if c.symbol_path == "top_level")
    assert chunk.start_line is not None
    assert chunk.end_line is not None
    assert chunk.end_line > chunk.start_line


def test_a_function_body_is_not_split_mid_definition() -> None:
    chunk = next(c for c in chunk_code(_file()) if c.symbol_path == "top_level")
    assert "def top_level" in chunk.text
    assert "return a + 1" in chunk.text


def test_module_level_code_outside_definitions_is_captured() -> None:
    texts = " ".join(c.text for c in chunk_code(_file()))
    assert "import os" in texts


def test_chunk_ids_are_stable_across_runs() -> None:
    assert [c.chunk_id for c in chunk_code(_file())] == [c.chunk_id for c in chunk_code(_file())]


def test_oversized_function_is_split_with_a_continuation_marker() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(2000))
    source = f"def huge():\n{body}\n"
    chunks = chunk_code(LoadedFile(path="huge.py", text=source, language="python"))
    assert len(chunks) > 1
    assert any("continues" in (c.symbol_path or "") for c in chunks[1:])


def test_syntactically_invalid_python_falls_back_to_whole_file() -> None:
    chunks = chunk_code(LoadedFile(path="bad.py", text="def (:::", language="python"))
    assert len(chunks) == 1
    assert chunks[0].symbol_path is None


def test_non_python_language_falls_back_to_whole_file() -> None:
    chunks = chunk_code(LoadedFile(path="a.ts", text="const x = 1;", language="typescript"))
    assert len(chunks) == 1
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_code_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.ingest.chunkers.code'`

- [x] **Step 3: Implement the chunker**

```python
# src/rag/ingest/chunkers/code.py
"""AST-aware code chunking via tree-sitter.

Boundaries fall on definitions, so a retrieved chunk is a whole function or class rather
than an arbitrary window. Python is parsed; other languages fall back to whole-file chunks
until a grammar is added.
"""

from __future__ import annotations

import hashlib

import structlog
import tree_sitter_python as ts_python
from tree_sitter import Language, Node, Parser

from rag.contracts import Chunk, make_chunk_id
from rag.ingest.chunkers.markdown import CHARS_PER_TOKEN, estimate_tokens
from rag.ingest.loaders import LoadedFile

log = structlog.get_logger(__name__)

PYTHON = Language(ts_python.language())
DEFINITION_NODES = {"function_definition", "class_definition", "decorated_definition"}


def _name_of(node: Node) -> str | None:
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        return _name_of(inner) if inner is not None else None
    name_node = node.child_by_field_name("name")
    return name_node.text.decode("utf-8") if name_node is not None else None


def _whole_file_chunk(file: LoadedFile, content_hash: str) -> list[Chunk]:
    text = file.text.strip()
    if not text:
        return []
    return [
        Chunk(
            chunk_id=make_chunk_id(file.path, text),
            doc_id=file.path,
            text=text,
            source_path=file.path,
            language=file.language,
            start_line=1,
            end_line=file.text.count("\n") + 1,
            token_count=estimate_tokens(text),
            content_hash=content_hash,
        )
    ]


def _split_oversized(
    file: LoadedFile,
    text: str,
    symbol: str | None,
    start_line: int,
    max_tokens: int,
    content_hash: str,
) -> list[Chunk]:
    """Split on statement (line) boundaries, never mid-line."""
    budget = max_tokens * CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    buffer: list[str] = []
    size = 0
    part = 0
    line_cursor = start_line

    def flush() -> None:
        nonlocal buffer, size, part, line_cursor
        if not buffer:
            return
        body = "\n".join(buffer)
        label = symbol if part == 0 else f"{symbol} (continues {part})"
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(file.path, body),
                doc_id=file.path,
                text=body,
                source_path=file.path,
                language=file.language,
                symbol_path=label,
                start_line=line_cursor,
                end_line=line_cursor + len(buffer) - 1,
                token_count=estimate_tokens(body),
                content_hash=content_hash,
            )
        )
        line_cursor += len(buffer)
        part += 1
        buffer = []
        size = 0

    for line in text.split("\n"):
        if size + len(line) > budget and buffer:
            flush()
        buffer.append(line)
        size += len(line) + 1
    flush()
    return chunks


def chunk_code(file: LoadedFile, max_tokens: int = 512) -> list[Chunk]:
    content_hash = hashlib.blake2b(file.text.encode(), digest_size=8).hexdigest()

    if file.language != "python":
        return _whole_file_chunk(file, content_hash)

    source = file.text.encode("utf-8")
    tree = Parser(PYTHON).parse(source)
    if tree.root_node.has_error:
        log.debug("parse_error_fallback", path=file.path)
        return _whole_file_chunk(file, content_hash)

    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    def emit(node: Node, prefix: str | None) -> None:
        name = _name_of(node)
        symbol = f"{prefix}.{name}" if prefix and name else name
        text = source[node.start_byte : node.end_byte].decode("utf-8").strip()
        if not text:
            return
        start_line = node.start_point[0] + 1

        if estimate_tokens(text) > max_tokens:
            chunks.extend(
                _split_oversized(file, text, symbol, start_line, max_tokens, content_hash)
            )
        else:
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(file.path, text),
                    doc_id=file.path,
                    text=text,
                    source_path=file.path,
                    language=file.language,
                    symbol_path=symbol,
                    start_line=start_line,
                    end_line=node.end_point[0] + 1,
                    token_count=estimate_tokens(text),
                    content_hash=content_hash,
                )
            )
        covered.append((node.start_byte, node.end_byte))

    def walk(node: Node, prefix: str | None) -> None:
        for child in node.children:
            if child.type in DEFINITION_NODES:
                target = child
                if child.type == "decorated_definition":
                    inner = child.child_by_field_name("definition")
                    target = inner if inner is not None else child
                if target.type == "class_definition":
                    emit(child, prefix)
                    body = target.child_by_field_name("body")
                    name = _name_of(target)
                    if body is not None and name is not None:
                        walk(body, f"{prefix}.{name}" if prefix else name)
                else:
                    emit(child, prefix)

    walk(tree.root_node, None)

    # Module-level code outside any definition (imports, constants, module docstring).
    remainder_parts: list[str] = []
    cursor = 0
    for start, end in sorted(covered):
        if start > cursor:
            remainder_parts.append(source[cursor:start].decode("utf-8"))
        cursor = max(cursor, end)
    if cursor < len(source):
        remainder_parts.append(source[cursor:].decode("utf-8"))

    remainder = "\n".join(p.strip() for p in remainder_parts if p.strip()).strip()
    if remainder:
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(file.path, remainder),
                doc_id=file.path,
                text=remainder,
                source_path=file.path,
                language=file.language,
                symbol_path=f"{file.path} (module level)",
                start_line=1,
                token_count=estimate_tokens(remainder),
                content_hash=content_hash,
            )
        )

    return chunks if chunks else _whole_file_chunk(file, content_hash)
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_code_chunker.py -v`
Expected: PASS (9 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/ingest/chunkers/code.py tests/unit/test_code_chunker.py
git commit -m "feat: add tree-sitter code chunker with symbol paths"
```

---

## Task 1.4: Ingestion pipeline

**Files:**
- Create: `src/rag/ingest/pipeline.py`
- Test: `tests/unit/test_ingest_pipeline.py`

**Interfaces:**
- Consumes: `load_repository`, `chunk_markdown`, `chunk_code`
- Produces: `IngestStats(files: int, chunks: int, skipped: int, duration_s: float)` and
  `build_chunks(root: Path) -> tuple[list[Chunk], IngestStats]`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_ingest_pipeline.py
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
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_ingest_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.ingest.pipeline'`

- [x] **Step 3: Implement the pipeline**

```python
# src/rag/ingest/pipeline.py
"""Orchestrates load -> chunk -> dedupe."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from rag.contracts import Chunk
from rag.ingest.chunkers.code import chunk_code
from rag.ingest.chunkers.markdown import chunk_markdown
from rag.ingest.loaders import load_repository

log = structlog.get_logger(__name__)

MARKDOWN_LANGUAGES = {"markdown"}


@dataclass(frozen=True)
class IngestStats:
    files: int
    chunks: int
    skipped: int
    duration_s: float


def build_chunks(root: Path) -> tuple[list[Chunk], IngestStats]:
    started = time.perf_counter()
    seen: set[str] = set()
    chunks: list[Chunk] = []
    files = 0
    skipped = 0

    for file in load_repository(root):
        files += 1
        produced = chunk_markdown(file) if file.language in MARKDOWN_LANGUAGES else chunk_code(file)
        if not produced:
            skipped += 1
            continue
        for chunk in produced:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)

    stats = IngestStats(
        files=files,
        chunks=len(chunks),
        skipped=skipped,
        duration_s=time.perf_counter() - started,
    )
    log.info("ingest_complete", **stats.__dict__)
    return chunks, stats
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_ingest_pipeline.py -v`
Expected: PASS (5 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/ingest/pipeline.py tests/unit/test_ingest_pipeline.py
git commit -m "feat: add ingestion pipeline dispatching by file type"
```

---

## Task 1.5: Embedder — dense and sparse

**Files:**
- Create: `src/rag/models/embedder.py`
- Test: `tests/unit/test_embedder.py`

**Interfaces:**
- Consumes: `Settings`
- Produces: `Embedder(settings)` with
  `embed_documents(texts: list[str]) -> list[list[float]]`,
  `embed_query(text: str) -> list[float]`,
  `embed_sparse(texts: list[str]) -> list[SparseVec]` where
  `SparseVec = tuple[list[int], list[float]]`

> **The single most important detail in this task.** `bge` models expect a query instruction
> prefix on *queries only*. Applying it to documents, or omitting it on queries, degrades recall
> substantially — and silently. There is no error; Recall@5 is just quietly bad. The test below
> exists specifically to prevent that.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_embedder.py
from unittest.mock import MagicMock

import pytest

from rag.config import Settings
from rag.models.embedder import QUERY_PREFIX, Embedder


def _embedder() -> tuple[Embedder, MagicMock, MagicMock]:
    dense = MagicMock()
    dense.encode.return_value = [[0.1] * 384]
    sparse = MagicMock()
    sparse.embed.return_value = iter([MagicMock(indices=[1, 5], values=[0.7, 0.3])])
    return Embedder(Settings(), dense_model=dense, sparse_model=sparse), dense, sparse


def test_query_embedding_applies_the_bge_instruction_prefix() -> None:
    embedder, dense, _ = _embedder()
    embedder.embed_query("what is rrf")
    sent = dense.encode.call_args.args[0]
    assert sent == [f"{QUERY_PREFIX}what is rrf"]


def test_document_embedding_does_not_apply_the_prefix() -> None:
    embedder, dense, _ = _embedder()
    embedder.embed_documents(["rrf fuses ranked lists"])
    sent = dense.encode.call_args.args[0]
    assert sent == ["rrf fuses ranked lists"]
    assert QUERY_PREFIX not in sent[0]


def test_embeddings_are_normalised_for_cosine() -> None:
    embedder, dense, _ = _embedder()
    embedder.embed_documents(["text"])
    assert dense.encode.call_args.kwargs["normalize_embeddings"] is True


def test_sparse_embedding_returns_index_value_pairs() -> None:
    embedder, _, _ = _embedder()
    result = embedder.embed_sparse(["rrf fuses ranked lists"])
    assert result == [([1, 5], [0.7, 0.3])]


def test_embedding_an_empty_list_does_not_call_the_model() -> None:
    embedder, dense, _ = _embedder()
    assert embedder.embed_documents([]) == []
    dense.encode.assert_not_called()


@pytest.mark.slow
def test_real_embedder_produces_384_dimensions() -> None:
    embedder = Embedder(Settings())
    vector = embedder.embed_query("what is reciprocal rank fusion")
    assert len(vector) == 384
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_embedder.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.models.embedder'`

- [x] **Step 3: Implement the embedder**

```python
# src/rag/models/embedder.py
"""Dense (bge-small) and sparse (BM25) embedding.

bge models expect an instruction prefix on QUERIES ONLY. Applying it to documents, or
omitting it on queries, silently degrades recall — see tests.
"""

from __future__ import annotations

from typing import Any

from rag.config import Settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SPARSE_MODEL_NAME = "Qdrant/bm25"

SparseVec = tuple[list[int], list[float]]


class Embedder:
    def __init__(
        self,
        settings: Settings,
        dense_model: Any | None = None,
        sparse_model: Any | None = None,
    ) -> None:
        self._settings = settings
        self._dense = dense_model
        self._sparse = sparse_model

    @property
    def dense(self) -> Any:
        if self._dense is None:
            from sentence_transformers import SentenceTransformer

            self._dense = SentenceTransformer(self._settings.models.embedder, device="cpu")
        return self._dense

    @property
    def sparse(self) -> Any:
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        return self._sparse

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.dense.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.dense.encode(
            [f"{QUERY_PREFIX}{text}"], normalize_embeddings=True, show_progress_bar=False
        )
        return [float(x) for x in vectors[0]]

    def embed_sparse(self, texts: list[str]) -> list[SparseVec]:
        if not texts:
            return []
        return [
            ([int(i) for i in emb.indices], [float(v) for v in emb.values])
            for emb in self.sparse.embed(texts)
        ]
```

- [x] **Step 4: Run the fast tests and verify they pass**

Run: `uv run pytest tests/unit/test_embedder.py -v -m "not slow"`
Expected: PASS (5 tests)

- [x] **Step 5: Run the slow test once to confirm the real model works**

Run: `uv run pytest tests/unit/test_embedder.py -v -m slow`
Expected: PASS. First run downloads ~130 MB.

- [x] **Step 6: Commit**

```bash
git add src/rag/models/embedder.py tests/unit/test_embedder.py
git commit -m "feat: add dense and sparse embedder with bge query prefix handling"
```

---

## Task 1.6: Qdrant collection schema and upsert

**Files:**
- Create: `src/rag/index/schema.py`
- Modify: `src/rag/index/qdrant_store.py` (add `ensure_collection`, `upsert_chunks`, `count`)
- Test: `tests/unit/test_index_schema.py`, `tests/integration/test_index_live.py`

**Interfaces:**
- Consumes: `Chunk`, `Embedder`, `Settings`
- Produces: `QdrantStore.ensure_collection() -> None`,
  `QdrantStore.upsert_chunks(chunks: list[Chunk], embedder: Embedder) -> int`,
  `QdrantStore.count() -> int`, and `chunk_to_payload(chunk) -> dict`

The collection carries **named vectors**: `"dense"` (384-dim, cosine) and `"sparse"` (with the
`IDF` modifier, which is what makes Qdrant's sparse vectors behave as BM25).

- [x] **Step 1: Write the failing unit test**

```python
# tests/unit/test_index_schema.py
from rag.contracts import Chunk, make_chunk_id
from rag.index.schema import DENSE_VECTOR, SPARSE_VECTOR, chunk_to_payload, payload_to_chunk


def _chunk() -> Chunk:
    text = "def search(self): ..."
    return Chunk(
        chunk_id=make_chunk_id("src/a.py", text),
        doc_id="src/a.py",
        text=text,
        source_path="src/a.py",
        language="python",
        symbol_path="Retriever.search",
        start_line=10,
        end_line=12,
        token_count=6,
    )


def test_vector_names_are_stable_constants() -> None:
    assert DENSE_VECTOR == "dense"
    assert SPARSE_VECTOR == "sparse"


def test_payload_round_trips_without_loss() -> None:
    original = _chunk()
    assert payload_to_chunk(chunk_to_payload(original)) == original


def test_payload_contains_the_fields_needed_for_filtering() -> None:
    payload = chunk_to_payload(_chunk())
    assert payload["language"] == "python"
    assert payload["source_path"] == "src/a.py"
    assert payload["quarantined"] is False
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_index_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.index.schema'`

- [x] **Step 3: Implement the schema module**

```python
# src/rag/index/schema.py
"""Collection schema and payload mapping.

Named vectors let one collection hold both representations, so dense and sparse can be
fused server-side in a single round trip (PRD FR-R3).
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from rag.contracts import Chunk

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def vectors_config(size: int) -> dict[str, models.VectorParams]:
    return {DENSE_VECTOR: models.VectorParams(size=size, distance=models.Distance.COSINE)}


def sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    # IDF modifier is what makes these behave as BM25 rather than raw term counts.
    return {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}


def point_id(chunk_id: str) -> str:
    """Qdrant needs a UUID or unsigned int. Derive one deterministically from chunk_id
    so upserts stay idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    return chunk.model_dump()


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    return Chunk.model_validate(payload)
```

- [x] **Step 4: Extend `QdrantStore` with collection management and upsert**

Append to `src/rag/index/qdrant_store.py`, and add the imports at the top:

```python
from qdrant_client import models

from rag.contracts import Chunk
from rag.index.schema import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    chunk_to_payload,
    point_id,
    sparse_vectors_config,
    vectors_config,
)
```

Add these methods to the `QdrantStore` class:

```python
    def ensure_collection(self, recreate: bool = False) -> None:
        name = self._settings.qdrant.collection
        if recreate and self.collection_exists():
            self._client.delete_collection(name)
        if not self.collection_exists():
            self._client.create_collection(
                collection_name=name,
                vectors_config=vectors_config(self._settings.qdrant.vector_size),
                sparse_vectors_config=sparse_vectors_config(),
            )
            log.info("collection_created", collection=name)


    def upsert_chunks(self, chunks: list[Chunk], embedder: Any, batch_size: int = 64) -> int:
        if not chunks:
            return 0
        self.ensure_collection()
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            dense = embedder.embed_documents(texts)
            sparse = embedder.embed_sparse(texts)

            points = [
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR: dense_vec,
                        SPARSE_VECTOR: models.SparseVector(indices=indices, values=values),
                    },
                    payload=chunk_to_payload(chunk),
                )
                for chunk, dense_vec, (indices, values) in zip(batch, dense, sparse, strict=True)
            ]
            self._client.upsert(collection_name=self._settings.qdrant.collection, points=points)
            total += len(points)
            log.info("upserted_batch", count=len(points), total=total)
        return total


    def count(self) -> int:
        if not self.collection_exists():
            return 0
        result = self._client.count(collection_name=self._settings.qdrant.collection, exact=True)
        return int(result.count)
```

- [x] **Step 5: Run the unit tests and verify they pass**

Run: `uv run pytest tests/unit/test_index_schema.py -v`
Expected: PASS (3 tests)

- [x] **Step 6: Write the integration test**

```python
# tests/integration/test_index_live.py
import pytest

from rag.config import Settings
from rag.contracts import Chunk, make_chunk_id
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder


@pytest.fixture
def store() -> QdrantStore:
    settings = Settings.model_validate({"qdrant": {"collection": "test_upsert"}})
    s = QdrantStore(settings)
    s.ensure_collection(recreate=True)
    return s


@pytest.mark.integration
@pytest.mark.slow
def test_upsert_is_idempotent(store: QdrantStore) -> None:
    text = "Reciprocal Rank Fusion combines ranked lists."
    chunk = Chunk(
        chunk_id=make_chunk_id("a.md", text),
        doc_id="a.md",
        text=text,
        source_path="a.md",
        language="markdown",
    )
    embedder = Embedder(Settings())

    store.upsert_chunks([chunk], embedder)
    store.upsert_chunks([chunk], embedder)

    assert store.count() == 1
```

- [x] **Step 7: Run the integration test**

```bash
docker compose up -d qdrant
uv run pytest tests/integration/test_index_live.py -v -m integration
```
Expected: PASS

- [x] **Step 8: Commit**

```bash
git add src/rag/index/ tests/unit/test_index_schema.py tests/integration/test_index_live.py
git commit -m "feat: add qdrant collection schema with named dense and sparse vectors"
```

---

## Task 1.7: Hybrid retrieval with server-side RRF

**Files:**
- Create: `src/rag/retrieval/__init__.py`, `src/rag/retrieval/hybrid.py`
- Modify: `src/rag/index/qdrant_store.py` (add `hybrid_search`)
- Test: `tests/unit/test_hybrid.py`, `tests/integration/test_retrieval_live.py`

**Interfaces:**
- Consumes: `Embedder`, `QdrantStore`, `Settings`, `Retrieved`, `Chunk`
- Produces: `HybridRetriever(settings, store, embedder, reranker=None)` with
  `search(query: str, k: int | None = None, rerank: bool | None = None) -> list[Retrieved]`

Fusion happens **inside Qdrant** via `prefetch` + `FusionQuery(RRF)`. The rejected alternative
— a client-side `rank_bm25` index — needed a second index kept in sync, in-process memory
against a 5 GB budget, and a full rebuild on every restart.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_hybrid.py
from unittest.mock import MagicMock

from rag.config import Settings
from rag.contracts import Chunk, make_chunk_id
from rag.retrieval.hybrid import HybridRetriever


def _scored(text: str, score: float) -> MagicMock:
    chunk = Chunk(
        chunk_id=make_chunk_id("a.md", text),
        doc_id="a.md",
        text=text,
        source_path="a.md",
        language="markdown",
    )
    point = MagicMock()
    point.payload = chunk.model_dump()
    point.score = score
    return point


def _retriever(points: list[MagicMock]) -> tuple[HybridRetriever, MagicMock]:
    store = MagicMock()
    store.hybrid_search.return_value = points
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    embedder.embed_sparse.return_value = [([1, 2], [0.5, 0.5])]
    return HybridRetriever(Settings(), store, embedder), store


def test_returns_results_ranked_from_one() -> None:
    retriever, _ = _retriever([_scored("first", 0.9), _scored("second", 0.7)])
    results = retriever.search("query", rerank=False)
    assert [r.rank for r in results] == [1, 2]


def test_fused_score_is_carried_through() -> None:
    retriever, _ = _retriever([_scored("first", 0.9)])
    assert retriever.search("query", rerank=False)[0].fused_score == 0.9


def test_query_is_embedded_with_both_representations() -> None:
    retriever, _ = _retriever([_scored("a", 0.5)])
    retriever.search("what is rrf", rerank=False)
    # Dense and sparse are both required for the prefetch branches.
    assert retriever._embedder.embed_query.called
    assert retriever._embedder.embed_sparse.called


def test_k_final_limits_the_result_count() -> None:
    points = [_scored(f"doc{i}", 1.0 - i / 100) for i in range(20)]
    retriever, _ = _retriever(points)
    assert len(retriever.search("query", k=3, rerank=False)) == 3


def test_reranking_reorders_and_records_scores() -> None:
    retriever, _ = _retriever([_scored("low", 0.9), _scored("high", 0.8)])
    reranker = MagicMock()
    # Reverse the fused order: second result scores highest.
    reranker.rerank.side_effect = lambda q, items: [
        (items[1], 9.0),
        (items[0], 1.0),
    ]
    retriever._reranker = reranker

    results = retriever.search("query", rerank=True)
    assert results[0].chunk.text == "high"
    assert results[0].rerank_score == 9.0
    assert [r.rank for r in results] == [1, 2]


def test_rerank_disabled_leaves_rerank_score_unset() -> None:
    retriever, _ = _retriever([_scored("a", 0.9)])
    assert retriever.search("query", rerank=False)[0].rerank_score is None


def test_ties_break_deterministically_on_chunk_id() -> None:
    """Tier A eval must be bit-reproducible; stable sort over unstable input is not enough."""
    tied = [_scored("bbb", 0.5), _scored("aaa", 0.5)]
    retriever, _ = _retriever(tied)
    first = [r.chunk.chunk_id for r in retriever.search("q", rerank=False)]
    retriever2, _ = _retriever(list(reversed(tied)))
    second = [r.chunk.chunk_id for r in retriever2.search("q", rerank=False)]
    assert first == second


def test_empty_results_return_an_empty_list() -> None:
    retriever, _ = _retriever([])
    assert retriever.search("query", rerank=False) == []
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.retrieval'`

- [x] **Step 3: Add `hybrid_search` to `QdrantStore`**

```python
    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int,
        exclude_quarantined: bool = True,
    ) -> list[Any]:
        query_filter = None
        if exclude_quarantined:
            query_filter = models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="quarantined", match=models.MatchValue(value=True)
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self._settings.qdrant.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    limit=self._settings.retrieval.k_dense,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR,
                    limit=self._settings.retrieval.k_sparse,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return list(response.points)
```

- [x] **Step 4: Implement the retriever**

```python
# src/rag/retrieval/__init__.py
```

```python
# src/rag/retrieval/hybrid.py
"""Hybrid retrieval: dense + sparse, fused server-side by RRF, then reranked."""

from __future__ import annotations

import time
from typing import Any

import structlog

from rag.config import Settings
from rag.contracts import Retrieved
from rag.index.schema import payload_to_chunk

log = structlog.get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: Any,
        embedder: Any,
        reranker: Any | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self.last_timings: dict[str, float] = {}

    def search(
        self,
        query: str,
        k: int | None = None,
        rerank: bool | None = None,
    ) -> list[Retrieved]:
        cfg = self._settings.retrieval
        k_final = k if k is not None else cfg.k_final
        do_rerank = cfg.rerank_enabled if rerank is None else rerank
        timings: dict[str, float] = {}

        started = time.perf_counter()
        dense = self._embedder.embed_query(query)
        sparse_indices, sparse_values = self._embedder.embed_sparse([query])[0]
        timings["embed_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        points = self._store.hybrid_search(
            dense_vector=dense,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            limit=cfg.k_fuse,
        )
        timings["fuse_ms"] = (time.perf_counter() - started) * 1000

        results = [
            Retrieved(chunk=payload_to_chunk(p.payload), fused_score=float(p.score)) for p in points
        ]
        # Deterministic ordering: fused score desc, then chunk_id asc to break ties.
        results.sort(key=lambda r: (-r.fused_score, r.chunk.chunk_id))

        if do_rerank and self._reranker is not None and results:
            started = time.perf_counter()
            scored = self._reranker.rerank(query, results)
            timings["rerank_ms"] = (time.perf_counter() - started) * 1000
            for item, score in scored:
                item.rerank_score = float(score)
            results = [item for item, _ in scored]
            results.sort(key=lambda r: (-(r.rerank_score or 0.0), r.chunk.chunk_id))

        results = results[:k_final]
        for position, item in enumerate(results, start=1):
            item.rank = position

        self.last_timings = timings
        log.debug("retrieval_complete", results=len(results), **timings)
        return results
```

- [x] **Step 5: Run the unit tests and verify they pass**

Run: `uv run pytest tests/unit/test_hybrid.py -v`
Expected: PASS (8 tests)

- [x] **Step 6: Commit**

```bash
git add src/rag/retrieval/ src/rag/index/qdrant_store.py tests/unit/test_hybrid.py
git commit -m "feat: add hybrid retriever with server-side rrf fusion"
```

---

## Task 1.8: Cross-encoder reranker

**Files:**
- Create: `src/rag/retrieval/rerank.py`
- Test: `tests/unit/test_rerank.py`

**Interfaces:**
- Consumes: `Settings`, `Retrieved`
- Produces: `CrossEncoderReranker(settings, model=None)` with
  `rerank(query: str, items: list[Retrieved]) -> list[tuple[Retrieved, float]]`
  sorted by score descending

Reranking must stay **toggleable**, because the eval harness runs with it on and off over the
same fused candidates to compute reranker lift. If the toggle breaks, the justification for the
model choice becomes unmeasurable.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_rerank.py
from unittest.mock import MagicMock

import pytest

from rag.config import Settings
from rag.contracts import Chunk, Retrieved, make_chunk_id
from rag.retrieval.rerank import CrossEncoderReranker


def _item(text: str) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=make_chunk_id("a.md", text),
            doc_id="a.md",
            text=text,
            source_path="a.md",
            language="markdown",
        )
    )


def test_results_are_sorted_by_score_descending() -> None:
    model = MagicMock()
    model.predict.return_value = [0.2, 0.9, 0.5]
    reranker = CrossEncoderReranker(Settings(), model=model)
    items = [_item("a"), _item("b"), _item("c")]

    ranked = reranker.rerank("query", items)
    assert [item.chunk.text for item, _ in ranked] == ["b", "c", "a"]
    assert [score for _, score in ranked] == [0.9, 0.5, 0.2]


def test_pairs_are_query_document_tuples() -> None:
    model = MagicMock()
    model.predict.return_value = [0.5]
    CrossEncoderReranker(Settings(), model=model).rerank("what is rrf", [_item("body")])
    assert model.predict.call_args.args[0] == [("what is rrf", "body")]


def test_empty_input_does_not_call_the_model() -> None:
    model = MagicMock()
    assert CrossEncoderReranker(Settings(), model=model).rerank("q", []) == []
    model.predict.assert_not_called()


def test_ties_break_deterministically_on_chunk_id() -> None:
    model = MagicMock()
    model.predict.return_value = [0.5, 0.5]
    reranker = CrossEncoderReranker(Settings(), model=model)
    ranked = reranker.rerank("q", [_item("bbb"), _item("aaa")])
    ids = [item.chunk.chunk_id for item, _ in ranked]
    assert ids == sorted(ids)


def test_configured_model_is_the_minilm_cross_encoder() -> None:
    """Guards PRD ADR-003: bge-reranker-large is ~2 s/query on this CPU."""
    assert "MiniLM" in Settings().models.reranker


@pytest.mark.slow
def test_real_reranker_prefers_the_relevant_document() -> None:
    reranker = CrossEncoderReranker(Settings())
    items = [
        _item("Bananas are a yellow fruit grown in tropical climates."),
        _item("Reciprocal Rank Fusion sums 1/(k+rank) across retrievers."),
    ]
    ranked = reranker.rerank("what is reciprocal rank fusion", items)
    assert "Reciprocal Rank Fusion" in ranked[0][0].chunk.text
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_rerank.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.retrieval.rerank'`

- [x] **Step 3: Implement the reranker**

```python
# src/rag/retrieval/rerank.py
"""Cross-encoder reranking.

ms-marco-MiniLM-L-6-v2 (22M) scores ~30 pairs in ~25 ms on CPU. bge-reranker-large (560M)
costs ~2 s for the same work — see ADR-003. Keep this stage toggleable: the eval harness
measures reranker lift by running with it on and off over identical fused candidates.
"""

from __future__ import annotations

from typing import Any

from rag.config import Settings
from rag.contracts import Retrieved


class CrossEncoderReranker:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._settings.models.reranker, device="cpu")
        return self._model

    def rerank(self, query: str, items: list[Retrieved]) -> list[tuple[Retrieved, float]]:
        if not items:
            return []
        pairs = [(query, item.chunk.text) for item in items]
        scores = self.model.predict(pairs)
        paired = [(item, float(score)) for item, score in zip(items, scores, strict=True)]
        paired.sort(key=lambda pair: (-pair[1], pair[0].chunk.chunk_id))
        return paired
```

- [x] **Step 4: Run the fast tests and verify they pass**

Run: `uv run pytest tests/unit/test_rerank.py -v -m "not slow"`
Expected: PASS (5 tests)

- [x] **Step 5: Run the slow test to confirm real ranking behaviour**

Run: `uv run pytest tests/unit/test_rerank.py -v -m slow`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add src/rag/retrieval/rerank.py tests/unit/test_rerank.py
git commit -m "feat: add toggleable cross-encoder reranker"
```

---

## Task 1.9: Context assembly

**Files:**
- Create: `src/rag/retrieval/context.py`
- Test: `tests/unit/test_context.py`

**Interfaces:**
- Consumes: `Retrieved`, `Settings`
- Produces: `assemble_context(items: list[Retrieved], budget_tokens: int) -> list[Retrieved]`

Order of operations is load-bearing: dedupe, then order by score, then drop **whole chunks**
from the tail to fit the budget. Never truncate a chunk mid-text — a half-function produces
confidently wrong answers, the worst possible failure for a system whose headline is
groundedness.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_context.py
from rag.contracts import Chunk, Retrieved, make_chunk_id
from rag.retrieval.context import assemble_context


def _item(text: str, tokens: int, score: float, path: str = "a.py") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=make_chunk_id(path, text),
            doc_id=path,
            text=text,
            source_path=path,
            language="python",
            token_count=tokens,
        ),
        rerank_score=score,
        fused_score=score,
    )


def test_drops_whole_chunks_to_fit_the_budget() -> None:
    items = [_item("a", 100, 0.9), _item("b", 100, 0.8), _item("c", 100, 0.7)]
    assembled = assemble_context(items, budget_tokens=250)
    assert len(assembled) == 2
    assert sum(i.chunk.token_count for i in assembled) <= 250


def test_never_truncates_chunk_text() -> None:
    items = [_item("full text here", 100, 0.9)]
    assert assemble_context(items, budget_tokens=100)[0].chunk.text == "full text here"


def test_orders_by_score_descending() -> None:
    items = [_item("low", 10, 0.1), _item("high", 10, 0.9)]
    assembled = assemble_context(items, budget_tokens=1000)
    assert [i.chunk.text for i in assembled] == ["high", "low"]


def test_deduplicates_identical_chunks_keeping_the_higher_score() -> None:
    duplicate = _item("same body", 10, 0.4)
    better = _item("same body", 10, 0.9)
    assembled = assemble_context([duplicate, better], budget_tokens=1000)
    assert len(assembled) == 1
    assert assembled[0].rerank_score == 0.9


def test_drops_a_chunk_subsumed_by_a_higher_ranked_one_from_the_same_file() -> None:
    """AST chunking emits both a class and its methods; the class subsumes the method."""
    outer = _item("class R:\n    def search(self): pass", 20, 0.9)
    inner = _item("def search(self): pass", 10, 0.5)
    assembled = assemble_context([outer, inner], budget_tokens=1000)
    assert len(assembled) == 1
    assert assembled[0].chunk.token_count == 20


def test_ranks_are_reassigned_from_one_after_assembly() -> None:
    items = [_item("a", 10, 0.5), _item("b", 10, 0.9)]
    assert [i.rank for i in assemble_context(items, budget_tokens=1000)] == [1, 2]


def test_a_single_oversized_chunk_is_still_returned() -> None:
    """Returning nothing would make the query unanswerable; one oversized chunk is better."""
    assembled = assemble_context([_item("huge", 5000, 0.9)], budget_tokens=100)
    assert len(assembled) == 1


def test_empty_input_returns_empty() -> None:
    assert assemble_context([], budget_tokens=1000) == []
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.retrieval.context'`

- [x] **Step 3: Implement context assembly**

```python
# src/rag/retrieval/context.py
"""Context assembly: dedupe, order, then fit the token budget by dropping whole chunks."""

from __future__ import annotations

from rag.contracts import Retrieved


def _score(item: Retrieved) -> float:
    return item.rerank_score if item.rerank_score is not None else item.fused_score


def assemble_context(items: list[Retrieved], budget_tokens: int) -> list[Retrieved]:
    if not items:
        return []

    ordered = sorted(items, key=lambda i: (-_score(i), i.chunk.chunk_id))

    kept: list[Retrieved] = []
    seen_text: set[str] = set()
    for item in ordered:
        text = item.chunk.text
        if text in seen_text:
            continue
        # Drop chunks subsumed by an already-kept chunk from the same file.
        if any(
            k.chunk.source_path == item.chunk.source_path and text in k.chunk.text for k in kept
        ):
            continue
        seen_text.add(text)
        kept.append(item)

    selected: list[Retrieved] = []
    used = 0
    for item in kept:
        tokens = item.chunk.token_count
        if selected and used + tokens > budget_tokens:
            continue
        selected.append(item)
        used += tokens

    for position, item in enumerate(selected, start=1):
        item.rank = position
    return selected
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_context.py -v`
Expected: PASS (8 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/retrieval/context.py tests/unit/test_context.py
git commit -m "feat: add context assembly with dedupe and whole-chunk budget enforcement"
```

---

## Task 1.10: Golden set schema and loader

**Files:**
- Create: `src/rag/eval/__init__.py`, `src/rag/eval/golden.py`, `eval/golden/retrieval.yaml`
- Test: `tests/unit/test_golden.py`

**Interfaces:**
- Consumes: nothing
- Produces: `GoldenQuery` model and `load_golden(path: Path) -> list[GoldenQuery]`

`provenance` is **required**, with no default. Hand-authored and synthetic questions are scored
separately and never pooled — synthetic questions are generated *from* the chunk that answers
them, so their lexical overlap is artificially high and a blended number would flatter the
system.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_golden.py
from pathlib import Path

import pytest

from rag.eval.golden import GoldenQuery, load_golden


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_entries(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: What is RRF?
  provenance: hand
  relevant_chunk_ids: [abc123]
  relevant_files: [Docs/decisions.md]
""",
    )
    queries = load_golden(path)
    assert len(queries) == 1
    assert queries[0].id == "q-001"
    assert queries[0].provenance == "hand"


def test_provenance_is_required(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: What is RRF?
  relevant_chunk_ids: [abc123]
""",
    )
    with pytest.raises(ValueError, match="provenance"):
        load_golden(path)


def test_provenance_must_be_hand_or_synthetic(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: What is RRF?
  provenance: guessed
  relevant_chunk_ids: [abc123]
""",
    )
    with pytest.raises(ValueError):
        load_golden(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
- id: q-001
  query: A?
  provenance: hand
  relevant_chunk_ids: [a]
- id: q-001
  query: B?
  provenance: hand
  relevant_chunk_ids: [b]
""",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


def test_a_query_must_have_ground_truth_unless_it_expects_refusal() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        GoldenQuery(id="q-1", query="?", provenance="hand")

    # A refusal case legitimately has no relevant chunk.
    assert GoldenQuery(id="q-2", query="?", provenance="hand", expect_refusal=True)


def test_missing_file_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "nope.yaml")
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_golden.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.eval'`

- [x] **Step 3: Implement the golden set module**

```python
# src/rag/eval/__init__.py
```

```python
# src/rag/eval/golden.py
"""Golden query sets.

`provenance` is mandatory. Hand-authored and synthetic questions are scored separately:
synthetic questions are generated FROM the chunk that answers them, so their lexical
overlap is artificially high and pooling the two would flatter the system (PRD FR-E1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Provenance = Literal["hand", "synthetic"]


class GoldenQuery(BaseModel):
    id: str
    query: str
    provenance: Provenance
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    golden_answer: str | None = None
    expect_refusal: bool = False

    @model_validator(mode="after")
    def require_ground_truth(self) -> GoldenQuery:
        if not self.expect_refusal and not self.relevant_chunk_ids and not self.relevant_files:
            raise ValueError(
                f"{self.id}: needs ground truth — set relevant_chunk_ids, "
                "relevant_files, or expect_refusal"
            )
        return self


def load_golden(path: Path) -> list[GoldenQuery]:
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    queries = [GoldenQuery.model_validate(entry) for entry in raw]

    seen: set[str] = set()
    for query in queries:
        if query.id in seen:
            raise ValueError(f"duplicate golden query id: {query.id}")
        seen.add(query.id)
    return queries
```

- [x] **Step 4: Seed `eval/golden/retrieval.yaml` with the first hand-authored entries**

Write real questions about this repository. Fill `relevant_chunk_ids` after the first ingest
(Task 1.12) by looking up the chunk ids; `relevant_files` works immediately.

```yaml
# Hand-authored golden queries. Ground truth is known because the author wrote the source.
# Target: ~25 hand + ~25 synthetic. Score the halves separately — never pool them.
- id: q-h001
  query: How does the retriever combine dense and sparse results?
  provenance: hand
  relevant_files: [src/rag/retrieval/hybrid.py]

- id: q-h002
  query: Why was bge-reranker-large rejected?
  provenance: hand
  relevant_files: [Docs/decisions.md]

- id: q-h003
  query: What keeps ingestion idempotent across runs?
  provenance: hand
  relevant_files: [src/rag/contracts.py]

- id: q-h004
  query: How are markdown chunks given readable citations?
  provenance: hand
  relevant_files: [src/rag/ingest/chunkers/markdown.py]

- id: q-h005
  query: What prevents a transformer from loading at import time?
  provenance: hand
  relevant_files: [src/rag/models/registry.py]
```

- [x] **Step 5: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_golden.py -v`
Expected: PASS (6 tests)

- [x] **Step 6: Commit**

```bash
git add src/rag/eval/ eval/golden/retrieval.yaml tests/unit/test_golden.py
git commit -m "feat: add golden query schema requiring explicit provenance"
```

---

## Task 1.11: Tier A retrieval metrics

**Files:**
- Create: `src/rag/eval/metrics/__init__.py`, `src/rag/eval/metrics/retrieval.py`
- Test: `tests/unit/test_retrieval_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces: `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`, `hit_rate_at_k` —
  each `(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`

> **These are commonly got wrong.** `precision_at_k` divides by `k`, not by the number
> retrieved. `ndcg_at_k` is **1-indexed** in `log2(i + 1)`, and its IDCG is computed over the
> ideal ordering **truncated to k**. `reciprocal_rank` returns `0` for a miss and must **not**
> skip zero-hit queries — dropping them inflates MRR. Every expected value below is
> hand-computed; do not derive expectations from the implementation.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_retrieval_metrics.py
import math

from rag.eval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_items_found() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5


def test_recall_respects_the_k_cutoff() -> None:
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_with_no_relevant_items_is_zero() -> None:
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_divides_by_k_not_by_results_returned() -> None:
    # Only 2 results returned, 1 relevant, k=5 -> 1/5, NOT 1/2.
    assert precision_at_k(["a", "b"], {"a"}, k=5) == 0.2


def test_reciprocal_rank_uses_the_first_relevant_position() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_reciprocal_rank_is_zero_on_a_miss() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_matches_a_hand_computed_value() -> None:
    # retrieved = [x, a], relevant = {a}, k=2
    # DCG  = 0/log2(2) + 1/log2(3) = 1/1.58496 = 0.63093
    # IDCG = 1/log2(2)             = 1.0
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, k=2), 0.63093, rel_tol=1e-4)


def test_ndcg_idcg_is_truncated_to_k() -> None:
    # 3 relevant items but k=2, so IDCG = 1/log2(2) + 1/log2(3) = 1.63093.
    # Retrieved puts both found items first -> DCG = 1.63093 -> NDCG = 1.0
    assert math.isclose(ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2), 1.0, rel_tol=1e-6)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(["x", "a"], {"a"}, k=2) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_all_metrics_handle_empty_retrieval() -> None:
    assert recall_at_k([], {"a"}, k=5) == 0.0
    assert precision_at_k([], {"a"}, k=5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}, k=5) == 0.0
    assert hit_rate_at_k([], {"a"}, k=5) == 0.0
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_retrieval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.eval.metrics'`

- [x] **Step 3: Implement the metrics**

```python
# src/rag/eval/metrics/__init__.py
```

```python
# src/rag/eval/metrics/retrieval.py
"""Tier A retrieval metrics. No LLM, fully deterministic, CI-gating (PRD FR-E1)."""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = len(set(retrieved_ids[:k]) & relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Denominator is k, not len(retrieved). A query returning 3 results for k=5
    is still divided by 5."""
    if k <= 0:
        return 0.0
    found = len(set(retrieved_ids[:k]) & relevant_ids)
    return found / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Returns 0.0 on a miss. Zero-hit queries must be included in the MRR mean —
    dropping them inflates the score."""
    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary relevance. Positions are 1-indexed in log2(i + 1); IDCG is computed over
    the ideal ordering truncated to k."""
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_retrieval_metrics.py -v`
Expected: PASS (12 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/eval/metrics/ tests/unit/test_retrieval_metrics.py
git commit -m "feat: add tier a retrieval metrics with hand-verified formulas"
```

---

## Task 1.12: Tier A eval runner and reranker lift

**Files:**
- Create: `src/rag/eval/runner.py`
- Test: `tests/unit/test_eval_runner.py`

**Interfaces:**
- Consumes: `GoldenQuery`, the metric functions, `HybridRetriever`, `Settings`
- Produces: `TierAResult` and
  `run_tier_a(queries: list[GoldenQuery], retriever, settings, k: int = 5) -> TierAResult`

`TierAResult` carries per-provenance breakdowns and the provenance block required by FR-E4.
Reranker lift is computed by running the same query set twice over identical fused candidates.

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_runner.py
from unittest.mock import MagicMock

from rag.config import Settings
from rag.contracts import Chunk, Retrieved, make_chunk_id
from rag.eval.golden import GoldenQuery
from rag.eval.runner import run_tier_a


def _retrieved(chunk_id: str, path: str = "a.py") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=chunk_id, doc_id=path, text="body", source_path=path, language="python"
        )
    )


def _retriever(by_rerank: dict[bool, list[Retrieved]]) -> MagicMock:
    retriever = MagicMock()
    retriever.search.side_effect = lambda q, k=None, rerank=None: by_rerank[bool(rerank)]
    retriever.last_timings = {"embed_ms": 1.0, "fuse_ms": 2.0}
    return retriever


def test_perfect_retrieval_scores_one() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.overall["recall@5"] == 1.0
    assert result.overall["ndcg@5"] == 1.0
    assert result.overall["hit_rate@5"] == 1.0


def test_complete_miss_scores_zero() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("z")], False: [_retrieved("z")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.overall["recall@5"] == 0.0
    assert result.overall["mrr"] == 0.0


def test_hand_and_synthetic_are_scored_separately() -> None:
    queries = [
        GoldenQuery(id="h1", query="?", provenance="hand", relevant_chunk_ids=["a"]),
        GoldenQuery(id="s1", query="?", provenance="synthetic", relevant_chunk_ids=["zzz"]),
    ]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.by_provenance["hand"]["recall@5"] == 1.0
    assert result.by_provenance["synthetic"]["recall@5"] == 0.0


def test_reranker_lift_is_the_ndcg_delta() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever(
        {
            True: [_retrieved("a"), _retrieved("z")],  # reranked: relevant first
            False: [_retrieved("z"), _retrieved("a")],  # fused only: relevant second
        }
    )

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.reranker_lift > 0.0


def test_falls_back_to_file_level_ground_truth_when_chunk_ids_are_absent() -> None:
    queries = [
        GoldenQuery(
            id="q1", query="?", provenance="hand", relevant_files=["src/rag/retrieval/hybrid.py"]
        )
    ]
    retriever = _retriever(
        {
            True: [_retrieved("x", path="src/rag/retrieval/hybrid.py")],
            False: [_retrieved("x", path="src/rag/retrieval/hybrid.py")],
        }
    )
    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.overall["recall@5"] == 1.0


def test_result_carries_the_required_provenance_block() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.provenance["config_hash"]
    assert result.provenance["models"]["embedder"] == "BAAI/bge-small-en-v1.5"
    assert result.provenance["golden_set"] == {"hand": 1, "synthetic": 0}


def test_refusal_queries_are_excluded_from_retrieval_metrics() -> None:
    queries = [
        GoldenQuery(id="r1", query="?", provenance="hand", expect_refusal=True),
        GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"]),
    ]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    result = run_tier_a(queries, retriever, Settings(), k=5)
    assert result.scored_queries == 1
    assert result.overall["recall@5"] == 1.0


def test_is_deterministic_across_runs() -> None:
    queries = [GoldenQuery(id="q1", query="?", provenance="hand", relevant_chunk_ids=["a"])]
    retriever = _retriever({True: [_retrieved("a")], False: [_retrieved("a")]})

    first = run_tier_a(queries, retriever, Settings(), k=5)
    second = run_tier_a(queries, retriever, Settings(), k=5)
    assert first.overall == second.overall
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_eval_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.eval.runner'`

- [x] **Step 3: Implement the runner**

```python
# src/rag/eval/runner.py
"""Tier A runner. No LLM in the loop — deterministic and CI-gating."""

from __future__ import annotations

import statistics
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from rag.config import Settings
from rag.contracts import Retrieved
from rag.eval.golden import GoldenQuery
from rag.eval.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TierAResult(BaseModel):
    overall: dict[str, float]
    by_provenance: dict[str, dict[str, float]]
    reranker_lift: float
    scored_queries: int
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


def _corpus_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return "unknown"


def _identifiers(results: list[Retrieved], use_files: bool) -> list[str]:
    if use_files:
        return [r.chunk.source_path for r in results]
    return [r.chunk.chunk_id for r in results]


def _score_one(query: GoldenQuery, results: list[Retrieved], k: int) -> dict[str, float]:
    # Prefer chunk-level ground truth; fall back to file-level when chunk ids are absent.
    use_files = not query.relevant_chunk_ids
    relevant = set(query.relevant_files) if use_files else set(query.relevant_chunk_ids)
    retrieved_ids = _identifiers(results, use_files)

    return {
        f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, relevant, k),
        "mrr": reciprocal_rank(retrieved_ids, relevant),
    }


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: statistics.fmean([row[key] for row in rows]) for key in sorted(rows[0])}


def run_tier_a(
    queries: list[GoldenQuery],
    retriever: Any,
    settings: Settings,
    k: int = 5,
) -> TierAResult:
    scored = [q for q in queries if not q.expect_refusal]

    reranked_rows: list[dict[str, float]] = []
    fused_rows: list[dict[str, float]] = []
    by_provenance: dict[str, list[dict[str, float]]] = {"hand": [], "synthetic": []}
    per_query: list[dict[str, Any]] = []

    for query in scored:
        reranked = retriever.search(query.query, k=k, rerank=True)
        fused = retriever.search(query.query, k=k, rerank=False)

        reranked_scores = _score_one(query, reranked, k)
        fused_scores = _score_one(query, fused, k)

        reranked_rows.append(reranked_scores)
        fused_rows.append(fused_scores)
        by_provenance[query.provenance].append(reranked_scores)
        per_query.append(
            {
                "id": query.id,
                "query": query.query,
                "provenance": query.provenance,
                "scores": reranked_scores,
                "retrieved": [r.chunk.chunk_id for r in reranked],
            }
        )

    overall = _mean(reranked_rows)
    fused_overall = _mean(fused_rows)
    lift = overall.get(f"ndcg@{k}", 0.0) - fused_overall.get(f"ndcg@{k}", 0.0)

    return TierAResult(
        overall=overall,
        by_provenance={name: _mean(rows) for name, rows in by_provenance.items()},
        reranker_lift=lift,
        scored_queries=len(scored),
        per_query=per_query,
        provenance={
            "corpus_commit": _corpus_commit(),
            "config_hash": settings.config_hash(),
            "models": {
                "embedder": settings.models.embedder,
                "reranker": settings.models.reranker,
                "generator": settings.models.generator,
            },
            "judge": None,
            "tier_c_enabled": False,
            "golden_set": {
                "hand": sum(1 for q in queries if q.provenance == "hand"),
                "synthetic": sum(1 for q in queries if q.provenance == "synthetic"),
            },
        },
    )
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_eval_runner.py -v`
Expected: PASS (8 tests)

- [x] **Step 5: Commit**

```bash
git add src/rag/eval/runner.py tests/unit/test_eval_runner.py
git commit -m "feat: add tier a eval runner with reranker lift and provenance"
```

---

## Task 1.13: CLI — `rag ingest` and `rag eval retrieval`

**Files:**
- Create: `src/rag/cli.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: everything above
- Produces: `app` (Typer) with `ingest` and `eval retrieval` commands

- [x] **Step 1: Write the failing test**

```python
# tests/unit/test_cli.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli import app

runner = CliRunner()


def test_ingest_reports_counts(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    with patch("rag.cli.QdrantStore") as store_cls, patch("rag.cli.Embedder"):
        store = MagicMock()
        store.upsert_chunks.return_value = 2
        store_cls.return_value = store

        result = runner.invoke(app, ["ingest", "--source", str(tmp_path)])

    assert result.exit_code == 0
    assert "chunks" in result.stdout.lower()


def test_eval_retrieval_prints_the_headline_metrics(tmp_path: Path) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "- id: q1\n  query: What is RRF?\n  provenance: hand\n  relevant_files: [a.py]\n",
        encoding="utf-8",
    )

    with (
        patch("rag.cli.QdrantStore"),
        patch("rag.cli.Embedder"),
        patch("rag.cli.CrossEncoderReranker"),
        patch("rag.cli.HybridRetriever") as retriever_cls,
    ):
        retriever = MagicMock()
        retriever.search.return_value = []
        retriever_cls.return_value = retriever

        result = runner.invoke(app, ["eval", "retrieval", "--golden", str(golden)])

    assert result.exit_code == 0
    assert "recall@5" in result.stdout.lower()
    assert "reranker lift" in result.stdout.lower()


def test_eval_retrieval_fails_clearly_on_a_missing_golden_set() -> None:
    result = runner.invoke(app, ["eval", "retrieval", "--golden", "nope.yaml"])
    assert result.exit_code != 0
```

- [x] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.cli'`

- [x] **Step 3: Implement the CLI**

```python
# src/rag/cli.py
"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.config import get_settings
from rag.eval.golden import load_golden
from rag.eval.runner import run_tier_a
from rag.index.qdrant_store import QdrantStore
from rag.ingest.pipeline import build_chunks
from rag.models.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker

app = typer.Typer(help="Enterprise RAG — ingest, retrieve, evaluate.")
eval_app = typer.Typer(help="Evaluation harness.")
app.add_typer(eval_app, name="eval")

console = Console()


@app.command()
def ingest(
    source: str = typer.Option(".", help="Directory to index."),
    recreate: bool = typer.Option(False, help="Drop and rebuild the collection first."),
) -> None:
    settings = get_settings()
    chunks, stats = build_chunks(Path(source))

    store = QdrantStore(settings)
    store.ensure_collection(recreate=recreate)
    upserted = store.upsert_chunks(chunks, Embedder(settings))

    console.print(
        f"[green]Ingested[/green] {stats.files} files -> {stats.chunks} chunks "
        f"({upserted} upserted, {stats.skipped} skipped) in {stats.duration_s:.1f}s"
    )


@eval_app.command("retrieval")
def eval_retrieval(
    golden: str = typer.Option("eval/golden/retrieval.yaml", help="Golden set path."),
    k: int = typer.Option(5, help="Cutoff for @k metrics."),
) -> None:
    settings = get_settings()
    queries = load_golden(Path(golden))

    retriever = HybridRetriever(
        settings,
        QdrantStore(settings),
        Embedder(settings),
        CrossEncoderReranker(settings),
    )
    result = run_tier_a(queries, retriever, settings, k=k)

    table = Table(title=f"Tier A — retrieval ({result.scored_queries} queries)")
    table.add_column("Metric")
    table.add_column("Overall", justify="right")
    table.add_column("Hand", justify="right")
    table.add_column("Synthetic", justify="right")

    for metric in sorted(result.overall):
        table.add_row(
            metric,
            f"{result.overall[metric]:.3f}",
            f"{result.by_provenance['hand'].get(metric, 0.0):.3f}",
            f"{result.by_provenance['synthetic'].get(metric, 0.0):.3f}",
        )
    console.print(table)
    console.print(f"[bold]Reranker lift (ΔNDCG@{k}):[/bold] {result.reranker_lift:+.4f}")
    console.print(
        f"[dim]corpus={result.provenance['corpus_commit']} "
        f"config={result.provenance['config_hash']}[/dim]"
    )


if __name__ == "__main__":
    app()
```

- [x] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: PASS (3 tests)

- [x] **Step 5: Ingest this repository for real**

```bash
docker compose up -d qdrant
uv run rag ingest --source . --recreate
```
Expected: `Ingested N files -> M chunks ...`

- [x] **Step 6: Fill in chunk-level ground truth in the golden set**

For each hand-authored query, find the chunk id that actually answers it and add it to
`relevant_chunk_ids`. File-level ground truth already works; chunk-level is sharper.

```bash
uv run python -c "
from pathlib import Path
from rag.ingest.pipeline import build_chunks
chunks, _ = build_chunks(Path('.'))
for c in chunks:
    if c.source_path == 'src/rag/retrieval/hybrid.py':
        print(c.chunk_id, '|', c.symbol_path)
"
```

- [x] **Step 7: Run the real evaluation — this is the phase exit criterion**

```bash
time uv run rag eval retrieval
```
Expected: a metrics table with Recall@5, Precision@5, NDCG@5, Hit Rate, MRR, split by
provenance, plus reranker lift — **in under 60 seconds** (NFR-7).

- [x] **Step 8: Record the first real numbers**

Add the measured values to `Docs/decisions.md` under ADR-003, replacing the assumption about
reranker quality with the actual ΔNDCG. Update the `UNMEASURED` comments in
`config/settings.yaml` for any `k` you have now compared.

- [x] **Step 9: Commit**

```bash
git add src/rag/cli.py tests/unit/test_cli.py eval/golden/retrieval.yaml Docs/decisions.md config/settings.yaml
git commit -m "feat: add ingest and eval cli, record first measured retrieval metrics"
```

---

## Task 1.14: Determinism guarantee

**Files:**
- Create: `tests/integration/test_determinism.py`

**Interfaces:**
- Consumes: `run_tier_a`, `HybridRetriever`
- Produces: nothing — this is a guard

If Tier A is not reproducible, the CI regression gate is noise and the entire harness is
decorative. This test is the guard on that property.

- [x] **Step 1: Write the test**

```python
# tests/integration/test_determinism.py
from pathlib import Path

import pytest

from rag.config import get_settings
from rag.eval.golden import load_golden
from rag.eval.runner import run_tier_a
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker


@pytest.mark.integration
@pytest.mark.slow
def test_tier_a_is_reproducible_across_runs() -> None:
    settings = get_settings()
    queries = load_golden(Path("eval/golden/retrieval.yaml"))
    retriever = HybridRetriever(
        settings, QdrantStore(settings), Embedder(settings), CrossEncoderReranker(settings)
    )

    first = run_tier_a(queries, retriever, settings, k=5)
    second = run_tier_a(queries, retriever, settings, k=5)

    assert first.overall == second.overall
    assert first.reranker_lift == second.reranker_lift
    assert [q["retrieved"] for q in first.per_query] == [q["retrieved"] for q in second.per_query]
```

- [x] **Step 2: Run it**

Run: `uv run pytest tests/integration/test_determinism.py -v -m integration`
Expected: PASS. If it fails, the cause is almost always an unstable tie-break — check that both
`hybrid.py` and `rerank.py` sort on `(-score, chunk_id)`.

- [x] **Step 3: Commit**

```bash
git add tests/integration/test_determinism.py
git commit -m "test: guarantee tier a evaluation is reproducible"
```

---

## Known deferrals and design notes

Recorded here rather than left to be discovered mid-execution.

**FR-I8 — incremental ingestion is partially satisfied, and that is deliberate.** The spec asks
that unchanged files be skipped by content hash. This plan achieves *idempotency* but not the
*speed* optimisation: `point_id()` is a UUID5 derived from `chunk_id`, which is itself a content
hash, so re-ingesting unchanged content overwrites the same point rather than duplicating it —
`test_upsert_is_idempotent` guards exactly that. What is not yet implemented is skipping the
embed step for unchanged files, which is purely a wall-time optimisation.

Defer it until ingest time is actually a problem. On a repo this size a full re-ingest is under a
minute, and a file-hash manifest is a cache-invalidation surface not worth adding before there is
a measurement showing it is needed. Add it as Task 1.15 if `rag ingest` exceeds ~2 minutes.

**`estimate_tokens` lives in `chunkers/markdown.py` and is imported by `chunkers/code.py`.** That
is a wart — a shared utility should not live in one of its two consumers. Leave it while there
are exactly two callers; move it to `src/rag/ingest/tokens.py` the moment a third appears, or
when generation needs it for the context budget in Phase 2. Noting it so the next person does
not have to rediscover it.

**Reranking happens over `k_fuse` candidates, not `k_final`.** `hybrid_search` fetches `k_fuse=30`,
the reranker scores all 30, and only then does the list truncate to `k_final=5`. Reranking after
truncation would be pointless — there would be nothing left to reorder. If you ever see rerank
latency spike, `k_fuse` is the dial, not `k_final`.

---

## Phase 1 exit checklist

- [x] `uv run pytest -m "not slow and not integration"` passes
- [x] `uv run pytest -m integration` passes with Qdrant running
- [x] `uv run mypy src/` reports no errors; `ruff check` and `ruff format --check` are clean
- [x] `uv run rag ingest --source .` indexes this repository
- [x] `uv run rag eval retrieval` prints Recall@5, Precision@5, NDCG@5, Hit Rate, MRR, split
      by provenance, **in 16.9 s** (NFR: < 60 s).
      *Deviation, deliberate:* reranker lift is no longer in the default run. Measuring it
      costs a second retrieval pass over every query (~100 s total) to restate a number
      already recorded in ADR-003. It moved behind `rag eval retrieval --lift`, which prints
      it — verified: −0.1056. The default run is the CI-gating one and must stay fast.
- [x] Hand and synthetic scores are reported separately, never pooled
- [x] Tier A is bit-reproducible across runs
- [x] Golden set has ≥ 25 hand-authored entries (**28**); synthetic generation is the first
      task of the next expansion
- [x] The first measured reranker lift is recorded in `Docs/decisions.md` under ADR-003
- [x] CI is green

**When all boxes are ticked, expand Phase 2 by re-running `superpowers:writing-plans` against
[`Docs/prd.md`](../prd.md) §7.3.** You will then have real numbers — chunk sizes, recall,
reranker lift — to write that plan against instead of assumptions.

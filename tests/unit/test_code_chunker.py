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

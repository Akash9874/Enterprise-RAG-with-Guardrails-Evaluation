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

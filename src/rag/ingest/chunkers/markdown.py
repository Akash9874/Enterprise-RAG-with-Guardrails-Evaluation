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

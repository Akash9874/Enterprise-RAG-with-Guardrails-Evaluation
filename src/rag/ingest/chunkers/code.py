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
    if name_node is None or name_node.text is None:
        return None
    return name_node.text.decode("utf-8")


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

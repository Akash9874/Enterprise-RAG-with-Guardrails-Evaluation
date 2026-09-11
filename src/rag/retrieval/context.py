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

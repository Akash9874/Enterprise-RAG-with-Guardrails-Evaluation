"""Synthetic golden queries (PRD §7.5). Ground truth by construction; easier by construction.

A question written *from* a chunk shares its vocabulary, so synthetic scores run high. That
is why they are scored separately and never pooled with hand-authored ones.
"""

from __future__ import annotations

import random
import re
from typing import Any

import yaml

from rag.contracts import Chunk
from rag.eval.golden import GoldenQuery

SYNTH_SYSTEM = (
    "You write one evaluation question about a passage from a software repository. The "
    "question must be answerable from the passage alone and must not quote file paths or "
    "identifiers verbatim. Reply in exactly this format and nothing else:\n"
    "QUESTION: <one question ending in ?>\n"
    "ANSWER: <one to three sentences using only the passage>"
)
_FORMAT = re.compile(r"QUESTION:\s*(?P<q>.+?)\s*ANSWER:\s*(?P<a>.*)", re.DOTALL)


def parse_generated(raw: str) -> tuple[str, str] | None:
    match = _FORMAT.search(raw)
    if match is None:
        return None
    question = " ".join(match["q"].split())
    answer = " ".join(match["a"].split())
    if not question.endswith("?") or not answer:
        return None
    return question, answer


def eligible_chunks(chunks: list[Chunk], min_tokens: int) -> list[Chunk]:
    """Sorted by chunk_id so the seeded shuffle is independent of Qdrant's scroll order."""
    keep = [c for c in chunks if c.token_count >= min_tokens and not c.quarantined]
    return sorted(keep, key=lambda c: c.chunk_id)


def synthesize(
    chunks: list[Chunk],
    llm: Any,
    n: int,
    seed: int,
    min_tokens: int,
    id_start: int = 1,
) -> tuple[list[GoldenQuery], int]:
    """Draft up to `n` accepted questions; return them and how many generations were rejected."""
    pool = eligible_chunks(chunks, min_tokens)
    random.Random(seed).shuffle(pool)
    queries: list[GoldenQuery] = []
    rejected = 0
    for chunk in pool:
        if len(queries) == n:
            break
        prompt = f'<passage path="{chunk.source_path}">\n{chunk.text}\n</passage>'
        parsed = parse_generated(llm.generate(prompt, system=SYNTH_SYSTEM))
        if parsed is None:
            rejected += 1
            continue
        question, answer = parsed
        queries.append(
            GoldenQuery(
                id=f"q-s{id_start + len(queries):03d}",
                query=question,
                provenance="synthetic",
                relevant_chunk_ids=[chunk.chunk_id],
                relevant_files=[chunk.source_path],
                golden_answer=answer,
                spot_checked=False,
            )
        )
    return queries, rejected


def dump_golden(queries: list[GoldenQuery]) -> str:
    rows = [q.model_dump(exclude_defaults=True) | {"spot_checked": q.spot_checked} for q in queries]
    return yaml.safe_dump(rows, sort_keys=False, allow_unicode=True, width=100)

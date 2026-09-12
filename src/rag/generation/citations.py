"""Citation construction and post-hoc enforcement (FR-G3, FR-G4).

A 3B model asked to cite will invent markers. Enforcement is *referential* only: a
surviving marker points at a chunk that was genuinely in context. Whether that chunk
supports the sentence is the Phase 3 groundedness rail's job, so `supported` stays None.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rag.contracts import Citation, Retrieved
from rag.generation.prompt import display_path

# A marker group: a single number, or several separated by commas, e.g. "[1]" or "[1, 4]".
_MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([.,;:!?)])")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")


class EnforcedCitations(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    stripped_markers: list[str] = Field(default_factory=list)
    ungrounded: bool = False


def build_citations(items: list[Retrieved]) -> list[Citation]:
    """Markers are positional and must match `build_prompt` exactly: `[1]` is the top item."""
    return [
        Citation(
            marker=f"[{position}]",
            chunk_id=item.chunk.chunk_id,
            source_path=item.chunk.source_path,
            display_path=display_path(item),
        )
        for position, item in enumerate(items, start=1)
    ]


def enforce_citations(text: str, citations: list[Citation]) -> EnforcedCitations:
    """Strip markers that do not resolve to a supplied source, and record each one."""
    by_marker = {citation.marker: citation for citation in citations}
    stripped: list[str] = []
    cited: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        kept: list[str] = []
        for number in (part.strip() for part in match.group(1).split(",")):
            marker = f"[{number}]"
            if marker in by_marker:
                kept.append(marker)
                cited.add(marker)
            else:
                stripped.append(marker)
        return "".join(kept)

    cleaned = _MARKER.sub(replace, text)
    cleaned = _REPEATED_SPACE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned).strip()

    used = [citation for citation in citations if citation.marker in cited]
    return EnforcedCitations(
        text=cleaned,
        citations=used,
        stripped_markers=stripped,
        ungrounded=not used,
    )

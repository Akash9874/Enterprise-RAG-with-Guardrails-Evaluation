"""Pure helpers for the demo UI. No streamlit import, so they are unit-testable."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

# One colour per verdict, in severity order. "block" is near-black on purpose: it was a red
# indistinguishable from "refuse", so a blocked request read as refused in the legend.
VERDICT_COLOURS: dict[str, str] = {
    "pass": "#2e7d32",
    "redact": "#1565c0",
    "hedge": "#ef6c00",
    "refuse": "#c62828",
    "block": "#212121",
    "skipped": "#9e9e9e",
    "error": "#6a1b9a",
}


def verdict_scale(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Colour-scale domain and range for only the verdicts on the chart.

    A fixed seven-entry legend was clipped to four in the UI, hiding "block" on a chart whose
    only bar was a block. Listing just the verdicts present keeps the legend truthful.
    """
    present = {row["verdict"] for row in rows}
    domain = [verdict for verdict in VERDICT_COLOURS if verdict in present]
    return domain, [VERDICT_COLOURS[verdict] for verdict in domain]


def api_url() -> str:
    return os.environ.get("RAG_UI_API_URL", "http://localhost:8000").rstrip("/")


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Decode `data:` lines of the /query event stream; ignore blanks and comments."""
    for line in lines:
        if line.startswith("data: "):
            yield json.loads(line[len("data: ") :])


def waterfall_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Rails laid end to end on a rail-time axis.

    Generation happens between the input and output rails but is excluded: a 30 s generation
    on the same axis would flatten 200 ms of rails into an invisible sliver.
    """
    rows: list[dict[str, Any]] = []
    offset = 0.0
    for stage, key in (("input", "input_rails"), ("output", "output_rails")):
        for rail in trace.get(key) or []:
            latency = float(rail.get("latency_ms") or 0.0)
            rows.append(
                {
                    "rail": rail["rail"],
                    "stage": stage,
                    "tier": rail["tier"],
                    "verdict": rail["verdict"],
                    "score": rail.get("score"),
                    "latency_ms": latency,
                    "start_ms": offset,
                    "end_ms": offset + latency,
                    "evidence": rail.get("evidence") or {},
                }
            )
            offset += latency
    return rows


def citation_panels(answer: dict[str, Any]) -> list[dict[str, str]]:
    """One expandable source panel per citation (FR-U2)."""
    chunks = {r["chunk"]["chunk_id"]: r["chunk"] for r in answer.get("retrieved") or []}
    panels: list[dict[str, str]] = []
    for citation in answer.get("citations") or []:
        chunk = chunks.get(citation["chunk_id"], {})
        lines = ""
        if chunk.get("start_line") is not None:
            end = chunk.get("end_line") or chunk["start_line"]
            lines = f":{chunk['start_line']}-{end}"
        panels.append(
            {
                "title": f"{citation['marker']} {citation['display_path']}",
                "location": f"{citation['source_path']}{lines}",
                "text": str(chunk.get("text", "")),
                "language": str(chunk.get("language", "text")),
            }
        )
    return panels


def load_presets(path: Path) -> list[tuple[str, str]]:
    """First case of each adversarial family, so a reviewer can trip the rails (FR-U4)."""
    seen: dict[str, str] = {}
    for case in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        seen.setdefault(case["family"], case["query"])
    return [(f"{family}: {query[:60]}", query) for family, query in sorted(seen.items())]

"""The rail contract (FR-GR2).

A rail returns a verdict; it never raises one. An exception means the rail malfunctioned,
and the pipeline converts that into `verdict="error"` plus the rail's fail-open or
fail-closed policy.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from rag.contracts import RailContext, RailResult, Tier


class Rail(Protocol):
    name: str
    tier: Tier

    def check(self, ctx: RailContext) -> RailResult: ...


@contextmanager
def timed() -> Iterator[list[float]]:
    """Rails time themselves — the pipeline does not time them (rail contract rule 3)."""
    holder = [0.0]
    started = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = (time.perf_counter() - started) * 1000

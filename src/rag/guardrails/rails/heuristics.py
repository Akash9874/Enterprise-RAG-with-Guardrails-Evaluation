"""T0 input heuristics: length, encoding, and a narrow denylist (FR-GR1).

Deliberately narrow. The corpus is source code, so a user asking "how do I ignore a file
in git?" is ordinary traffic — a broad `ignore` pattern would refuse real questions all
day. Every pattern here requires an override verb *and* an instruction-like object, and
each has a false-positive control in the tests.

This rail costs microseconds and runs first, so a blatant attack never reaches the 120 ms
classifier behind it.
"""

from __future__ import annotations

import re

from rag.contracts import RailContext, RailResult, Tier, Verdict
from rag.guardrails.base import timed
from rag.guardrails.policy import RailPolicy

# Control and zero-width characters. Tab, newline and carriage return are excluded: they
# are ordinary in a question about code. The rest are used to smuggle instructions past
# pattern matching, and have no legitimate place in a query.
_SUSPICIOUS_CHARS = re.compile(
    r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u200b-\u200f\u2028\u2029\ufeff]"
)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.!?\n]{0,40}?"
            r"\b(?:instruction|prompt|rule|direction|guideline)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(?:reveal|print|show|repeat|output|disclose|leak|dump)\b[^.!?\n]{0,30}?"
            r"\bsystem\s+prompt\b",
            re.IGNORECASE,
        ),
    ),
    (
        "persona_override",
        re.compile(
            r"\byou\s+are\s+now\b[^.!?\n]{0,30}?"
            r"\b(?:dan|developer\s+mode|jailbroken|jailbreak|unrestricted|admin|root)\b",
            re.IGNORECASE,
        ),
    ),
)


class InputHeuristicsRail:
    name = "input_heuristics"
    tier: Tier = "T0"

    def __init__(self, policy: RailPolicy) -> None:
        self._policy = policy

    def check(self, ctx: RailContext) -> RailResult:
        with timed() as elapsed:
            verdict, evidence = self._evaluate(ctx.query)
        return RailResult(
            rail=self.name,
            tier="T0",
            verdict=verdict,
            latency_ms=elapsed[0],
            evidence=evidence,
        )

    def _evaluate(self, query: str) -> tuple[Verdict, dict[str, object]]:
        # Cheapest and least ambiguous checks first.
        trip: Verdict = self._policy.trip_verdict  # declared in policy, not hardcoded (FR-GR3)

        if len(query) > self._policy.max_query_chars:
            return trip, {"reason": "too_long", "length": len(query)}

        smuggled = _SUSPICIOUS_CHARS.search(query)
        if smuggled is not None:
            return trip, {
                "reason": "suspicious_encoding",
                "codepoint": f"U+{ord(smuggled.group()):04X}",
                "offset": smuggled.start(),
            }

        for name, pattern in _PATTERNS:
            match = pattern.search(query)
            if match is not None:
                return trip, {
                    "reason": "denylist",
                    "matched_pattern": name,
                    "span": [match.start(), match.end()],
                    "matched_text": match.group()[:120],
                }

        return "pass", {"reason": "no_match", "checked": len(_PATTERNS)}

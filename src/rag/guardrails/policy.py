"""Declarative guardrail policy (FR-GR3).

Thresholds live in `config/guardrails.yaml`, never in source. Every one of them should
end up justified by a number from the eval harness; until then they carry an UNMEASURED
comment in the YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from rag.config import _project_root
from rag.contracts import Verdict

Action = Literal["allow", "redact", "hedge", "refuse", "block"]
OnError = Literal["open", "closed"]

DEFAULT_POLICY_PATH = _project_root() / "config" / "guardrails.yaml"


class RailPolicy(BaseModel):
    enabled: bool = True
    action: Action = "block"
    # Safety rails deny on malfunction, quality rails degrade. Flipping either changes
    # the project's safety posture and belongs in the PRD, not in a config tweak.
    on_error: OnError = "closed"
    t_pass: float | None = None
    t_block: float | None = None
    escalate: bool = False
    entities: list[str] = Field(default_factory=list)
    max_query_chars: int = 4000

    @property
    def trip_verdict(self) -> Verdict:
        """The verdict a rail returns when it trips — declared here, never hardcoded.

        `allow` maps to `pass`: a policy can neutralise a rail without disabling it, so
        the rail still runs and still reports its score to the trace.
        """
        return "pass" if self.action == "allow" else self.action

    @property
    def band(self) -> tuple[float, float] | None:
        """The interval where the rail admits it does not know, as (low, high).

        Deliberately direction-agnostic. Rails score in two opposite orientations and
        both are in the shipped policy: injection is a *risk* score where higher is
        worse (t_pass 0.30 < t_block 0.80), groundedness a *confidence* score where
        higher is better (t_block 0.35 < t_pass 0.65). Ordering the thresholds here
        would encode one orientation and silently mis-band the other, so each rail
        interprets its own direction and this property only reports the interval.
        """
        if self.t_pass is None or self.t_block is None:
            return None
        low, high = sorted((self.t_pass, self.t_block))
        return (low, high)


class GuardrailPolicy(BaseModel):
    escalation_budget_ms: int = 5000
    rails: dict[str, RailPolicy] = Field(default_factory=dict)

    def for_rail(self, name: str) -> RailPolicy:
        return self.rails.get(name, RailPolicy())


def load_policy(path: Path | None = None) -> GuardrailPolicy:
    """Read the policy file.

    A missing file raises rather than defaulting: a silently defaulted safety policy is
    indistinguishable from one that disabled a rail by accident.
    """
    resolved = path if path is not None else DEFAULT_POLICY_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"guardrail policy not found: {resolved}")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return GuardrailPolicy.model_validate(data)

# Phase 3 — The Tiered Guardrail Pipeline

**Goal:** Four rail families, ordered by cost, short-circuiting on a terminal verdict, with every
decision recorded in an inspectable trace — and an adversarial suite that reports how well they
actually work, including where they do not.

**Spec:** [`Docs/prd.md`](../prd.md) §7.4 · **Module guide:** [`src/rag/guardrails/CLAUDE.md`](../../src/rag/guardrails/CLAUDE.md)

**Exit criteria:** Adversarial suite runs; attack-success and false-refusal rates are reported;
guardrail overhead p50 ≤ 300 ms.

---

## What shipped

| Task | Deliverable | Spec |
|---|---|---|
| 3.1 | `Rail` protocol, `RailContext`, cost-ordered pipeline with short-circuit | FR-GR1, FR-GR2 |
| 3.2 | Policy loader — `config/guardrails.yaml`, thresholds, fail-open/closed | FR-GR3, FR-GR5 |
| 3.3 | T0 input heuristics — regex, denylist, length, encoding | FR-GR1 |
| 3.4 | T0 PII rail (Presidio) — input redaction and output leak re-scan | FR-GR1 |
| 3.5 | T1 injection rail — `deberta-v3-base-prompt-injection-v2` | FR-GR1 |
| 3.6 | T1 topicality rail — cosine to corpus centroid | FR-GR1 |
| 3.7 | T2 groundedness rail — HHEM per sentence per chunk | FR-GR1 |
| 3.8 | T3 escalation — LLM self-check, band only, budget-capped | FR-GR4, FR-GR7 |
| 3.9 | `GuardrailTrace` assembly and `include_trace` on the API | FR-GR6 |
| 3.10 | Ingest-time injection scan + `quarantined` flag | FR-I7 |
| 3.11 | Adversarial suite — 34 cases including 12 benign controls | FR-E5 |
| 3.12 | `rag bench` asserting guardrail overhead p50 | NFR-2 |

---

## Exit criteria, verified 2026-09-12

| Criterion | Result |
|---|---|
| Adversarial suite runs | 34 cases, `rag eval adversarial [--full]` |
| Attack success and false refusal both reported | **22.7%** and **0.0%** |
| Guardrail overhead p50 ≤ 300 ms (NFR-2) | **184 ms p50**, 220 ms p95 — PASS |
| Peak resident memory ≤ 6 GB (NFR-4) | **≈ 3.72 GB** — PASS |
| Coverage ≥ 85% on `guardrails/`, `retrieval/`, `eval/` (NFR-8) | **97%** — PASS |

### Attack success by family (full path, 34 cases)

| Family | Cases | Got through | Rate |
|---|---|---|---|
| injection | 8 | 0 | **0.0%** |
| pii | 4 | 0 | **0.0%** |
| out_of_scope | 6 | 1 | 16.7% |
| unanswerable | 4 | 4 | **100.0%** |

Six of the eight injection attempts are stopped by the T0 denylist in microseconds; only two
reach the 120 ms classifier. That is the tiering paying for itself, and it is why a 120 ms
classifier fits inside a 300 ms budget.

---

## The honest part: what does not work

**`unanswerable` is 100% through, and that is a deliberate policy choice, not an oversight.**
Groundedness is configured `action: hedge`, so a plausible-sounding question the corpus cannot
answer gets an answer carrying a warning rather than a refusal. The suite counts a hedged answer
as "got through", because the user still receives it — scoring it as stopped would flatter the
number.

Setting `action: refuse` moves the trade, and the cost is quantified rather than guessed:

| groundedness action | attack success | false refusal |
|---|---|---|
| `hedge` (shipped) | 22.7% | **0.0%** |
| `refuse` | ~9% | ~20% |

Hedging was chosen because false refusals are the failure a reviewer actually experiences, and
because a quality rail denying service contradicts FR-GR5. The alternative is one line of YAML.

**The first full-path run refused 11 of 12 benign controls** (91.7% false refusal) while
reporting excellent attack numbers. Two compounding defects — hardcoded verdicts (ADR-022) and
uncalibrated thresholds (ADR-021) — and only the benign half of the suite could tell.

**`US_SSN` is configured but does not fire** on a bare `123-45-6789`; Presidio wants surrounding
context. Recorded in ADR-020 rather than left implied by the config.

---

## Decisions recorded this phase

| ADR | Subject |
|---|---|
| [ADR-017](../decisions.md) | Pin `transformers < 5` — HHEM's remote code does not load on 5.x |
| [ADR-018](../decisions.md) | Measured model budget; PRD §4.1 was optimistic in both directions |
| [ADR-019](../decisions.md) | Topicality thresholds from measured separation — closes ADR-015's gap |
| [ADR-020](../decisions.md) | PII threshold, and redaction output tripping the next rail |
| [ADR-021](../decisions.md) | Groundedness per chunk with max; HHEM's 512-token window |
| [ADR-022](../decisions.md) | Trip verdicts come from policy, never hardcoded |

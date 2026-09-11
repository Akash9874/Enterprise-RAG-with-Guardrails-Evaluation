# CLAUDE.md — `guardrails/`

The tiered guardrail pipeline. This is the project's headline feature; treat its invariants as
load-bearing. Product requirements: [PRD §7.4](../../../Docs/prd.md).

---

## The core idea

Safety checks are ordered by **cost**, not by importance. Cheap checks run always; expensive
checks run only when cheap ones are uncertain.

| Tier | Cost | Mechanism | Runs |
|---|---|---|---|
| **T0** | microseconds | Regex, denylists, length/encoding, Presidio | always |
| **T1** | ~10–40 ms | Small encoder classifiers, embedding distance | always |
| **T2** | ~150 ms | HHEM cross-encoder NLI, structural checks | always (output) |
| **T3** | ~2–4 s | LLM self-check | **escalation only** |

The naive design — one LLM call per rail — costs 60+ seconds per query on this hardware. The
tiering is what makes the feature possible at all, and it is the thing worth explaining to a
reviewer.

**T3 is reachable from exactly one rail: output groundedness.** Input rails resolve on
deterministic logic or classifier score alone. Do not add T3 escalation to another rail without
changing PRD §7.4 FR-GR4 first.

---

## The escalation band

Each rail has two thresholds. The interval between them is where the rail admits it does not
know.

```
  0.0 ─────────── t_pass ────────────── t_block ─────────── 1.0
       clearly OK   │   ESCALATION BAND    │   clearly bad
       → pass       │   → T3 LLM check     │   → trip action
```

Tuning consequence: **widening the band raises accuracy and raises latency.** The T3 escalation
rate is a tracked NFR (≤ 10% of golden-set queries, NFR-3). If a threshold change pushes
escalation above that, it is a regression even if accuracy improved — report both numbers.

`escalation_budget_ms` caps total T3 work per request. On exhaustion, the rail degrades to its
T2 verdict and sets `budget_exceeded=true` on the trace. Never silently skip the budget check.

---

## Rail contract

Every rail is a pure-ish function returning a `RailResult`. No rail mutates the request in
place, writes global state, or raises to signal a verdict.

```python
class Rail(Protocol):
    name: str
    tier: Literal["T0", "T1", "T2", "T3"]

    def check(self, ctx: RailContext) -> RailResult: ...
```

Rules:

1. **Return, never raise, for a verdict.** Exceptions mean the rail malfunctioned, and are
   caught by the pipeline and converted to `verdict="error"` plus the fail-open/fail-closed
   policy for that rail.
2. **Always populate `evidence`.** Matched spans, offending patterns, unsupported sentences.
   A verdict without evidence is unreviewable, and the UI trace panel renders this field.
3. **Always populate `latency_ms`**, measured inside the rail. The pipeline does not time you.
4. **Redaction is recorded, never silent.** If a rail rewrites text, the original span and the
   replacement both go into `evidence`.
5. **Rails are independently testable** — pure inputs, deterministic outputs given a fixed
   model. No rail may depend on another rail having run.

---

## Fail-open vs fail-closed

On rail **error** (not on a trip):

| Rail family | Behaviour | Rationale |
|---|---|---|
| PII, injection | **fail closed** — deny | A safety rail that is down must not silently stop protecting. |
| Groundedness, topicality | **fail open** — allow, with a hedge | A quality rail being down should degrade the answer, not the service. |

Both are overridable per-rail in policy. The default must remain as above — if you flip one,
you are changing the project's safety posture, which belongs in the PRD.

---

## Policy

`config/guardrails.yaml`. Declarative, hot-reloadable, never hardcoded in source.

```yaml
escalation_budget_ms: 5000

rails:
  pii_input:
    enabled: true
    action: redact          # allow | redact | hedge | refuse | block
    on_error: closed
    entities: [EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IBAN_CODE, US_SSN]
    t_block: 0.5

  injection_input:
    enabled: true
    action: block
    on_error: closed
    t_pass: 0.30
    t_block: 0.80

  topicality:
    enabled: true
    action: refuse
    on_error: open
    t_pass: 0.45            # cosine distance to corpus centroid

  groundedness:
    enabled: true
    action: hedge
    on_error: open
    t_pass: 0.65            # HHEM factual-consistency score
    t_block: 0.35
    escalate: true          # the ONLY rail permitted to set this
```

Every threshold in this file should ultimately be justified by a number from the eval harness.
Thresholds chosen by intuition are placeholders and should be marked as such in a comment until
measured.

---

## Adding a rail

1. Add the class in `rails/`, implementing the `Rail` protocol.
2. Register it in `pipeline.py` **at the correct tier position** — cost order is the whole
   design; a misplaced rail silently destroys the latency budget.
3. Add its policy block to `config/guardrails.yaml` with explicit thresholds.
4. **Write the failing test first.** Rails are pure functions over fixed inputs; there is no
   excuse for an untested one.
5. Add at least two adversarial cases to `eval/adversarial/suite.yaml` — one the rail should
   catch, one it should *not* (false-positive control).
6. Re-run `rag bench` and confirm guardrail overhead p50 is still ≤ 300 ms (NFR-2).

---

## Indirect prompt injection

The corpus is source code and documentation, so a *retrieved chunk* can carry adversarial
instructions. This is the harder attack and gets layered defence:

1. **Ingest-time scan** — chunks are classified on the way in; those above `t_block` are
   flagged `quarantined=true` and excluded from retrieval by default (FR-I7). Catching it here
   is far cheaper than catching it at query time.
2. **Structural isolation** — retrieved context is wrapped in a delimited block explicitly
   labelled as data that must never be interpreted as instructions (FR-G2).
3. **Output topic-drift check** — the answer's embedding is compared against the query and
   retrieved context. A successful injection usually pulls the answer off-topic.

None of these is sufficient alone. Do not remove one because another exists.

---

## Testing rails

- Assert on the **`RailResult`** — verdict, tier, evidence keys. Never on log output.
- **Never assert an exact model score.** Assert the verdict and the band it fell into. Model
  outputs drift across versions; verdicts are the contract.
- Every rail needs a **false-positive test**: benign input that superficially resembles an
  attack. `"How do I ignore a file in git?"` must not trip the injection rail.
- Pipeline tests assert **ordering and short-circuiting** — that a T0 block prevents T1 from
  running at all. This is a correctness property, not an optimisation.
- Mark model-loading tests `@pytest.mark.slow`.

# Enterprise RAG with Guardrails & Evaluation

[![CI](https://github.com/Akash9874/Enterprise-RAG-with-Guardrails-Evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/Akash9874/Enterprise-RAG-with-Guardrails-Evaluation/actions/workflows/ci.yml)

A Retrieval-Augmented Generation service that answers questions about its own codebase, **runs
entirely on a CPU-only laptop at zero marginal cost**, and wraps every request in a cost-tiered
guardrail pipeline whose every decision is inspectable. Every quality claim below comes from the
project's own evaluation harness — including the unflattering ones, of which there are several.

> **Status: all five phases (0–4) complete.** Both Phase 4 exit criteria are met and verified: CI
> fails on a deliberately injected retrieval regression, and `docker compose up` reaches a healthy
> demo in **21.4 s**.

![A prompt-injection attempt blocked by the cheapest guardrail rail in under a millisecond](Docs/images/guardrail-block.png)

*A prompt-injection attempt, stopped by the T0 heuristics rail in **0.92 ms**. The 120 ms injection
classifier never runs. Ordering rails by cost is what makes guardrails affordable on a CPU.*

---

## Why it is shaped this way

It was built under one hard constraint: **a laptop with no usable GPU, 16 GB of RAM, and no budget
for paid inference.**

| | |
|---|---|
| CPU | AMD Ryzen 5 7530U, 6C/12T |
| GPU | None usable (integrated Radeon, ~0.5 GB) |
| RAM | 15.4 GB — model budget capped at ~5 GB |
| Cost | Zero marginal. No paid API in any default path |

That constraint forces real decisions. **Guardrails can't call an LLM per rail** — four sequential
LLM checks is a minute per query here — so rails are tiered by cost, and an LLM self-check is
reachable only from one rail's uncertainty band. **Evaluation can't lean on a judge model** — a
local 3B judge correlates poorly with people — so the backbone is zero-LLM and deterministic, with
LLM-judged metrics as an opt-in tier. **Several popular libraries didn't fit**, and each rejection is
recorded with its evidence in [`Docs/decisions.md`](Docs/decisions.md).

## What this demonstrates

- **Cost-tiered guardrails with an inspectable trace.** T0 regex and PII checks, T1 encoder
  classifiers, T2 NLI groundedness, and a T3 LLM check reachable only from one rail's escalation band.
- **A zero-LLM evaluation backbone that is actually deterministic.** On the same commit, the laptop and
  the CI runner produce identical Tier A metrics to four decimals, and all 28 queries return the same
  top-5 order.
- **A regression gate proven to fail.** An injected retrieval regression turned CI red:
  [run 34758239606](https://github.com/Akash9874/Enterprise-RAG-with-Guardrails-Evaluation/actions/runs/34758239606),
  Recall@5 0.661 → 0.464.
- **Provenance that can't lie about its corpus.** Every eval report records the corpus commit read back
  from the index itself, not from whichever checkout happened to run the eval.
- **Measurement over assertion.** 30 architecture decision records — several documenting a confident
  assumption that lost to a number.

---

## Measured results

All figures from the reference laptop and this repository's CI, 2026-09-12 to 2026-09-14. The golden
set is **hand-authored only** (28 answerable + 6 refusal queries); a synthetic half is built but not yet
merged.

**Retrieval and the CI gate**

| | |
|---|---|
| Tier A, 28 queries, dense + sparse RRF | Recall@5 **0.643** · NDCG@5 **0.665** · MRR **0.607** · Hit@5 **0.714** |
| Evaluation fixtures accidentally indexed as corpus, then excluded | cost **−0.034 NDCG@5** while indexed, with recall unchanged ([ADR-023](Docs/decisions.md)) |
| Gate on an injected regression | Recall@5 0.661 → **0.464**, **FAIL**, as designed ([ADR-026](Docs/decisions.md)) |
| Laptop vs CI runner, same commit | **identical**; 0 of 28 top-5 orderings differ |

**Generation — Tier B, zero-LLM scoring, 34 hand queries (98.5 min on CPU)**

| | |
|---|---|
| BERTScore F1 vs golden answers | **0.827** |
| Citation precision / recall | **0.833** / **0.205** |
| Answers citing nothing | **57%** (16 of 28) |
| Groundedness (HHEM, per cited chunk) | **0.168** — low mainly because uncited sentences score 0 |
| Refusal correctness | **0.941** — 0 false refusals, 2 missed refusals |
| Tier C, LLM-judged, **n = 3**, local 3B judge | faithfulness 0.500 · answer relevancy 0.328 · ~158 s/query ([ADR-027](Docs/decisions.md)) |

**Guardrails and adversarial robustness**

| | |
|---|---|
| Attack success rate, 34 cases (Phase 3 run) | **22.7%** — injection 0/8, PII 0/4, out-of-scope 1/6, unanswerable 4/4 |
| False refusal rate, 12 benign controls | **0.0%** |
| Input-rail overhead | **184 ms p50**, 220 ms p95 |
| Groundedness rail | **~14 s** per typical answer natively, 43 s observed in Docker — NFR-2 **not met** ([ADR-029](Docs/decisions.md)) |

**Demo, packaging and engineering**

| | |
|---|---|
| `docker compose up -d --wait` to healthy | **21.4 s** (target ≤ 90 s) |
| Docker image | **3.12 GB** CPU-only — down from 5.46 GB ([ADR-028](Docs/decisions.md)) |
| Resident model memory | **≈ 3.7 GB** (target ≤ 6 GB) |
| Tests | **442** fast-suite tests · `mypy --strict` clean · **94%** coverage on `guardrails/`, `retrieval/`, `eval/` |

---

## The guardrail trace

![A correctly cited answer with its full six-rail guardrail trace and a T3 escalation](Docs/images/guardrail-trace-escalation.png)

*A correctly cited answer (`[2]` → ADR-003) that the groundedness rail hedged. Its score, 0.309, fell
inside the escalation band, so it escalated to a T3 LLM check — which ran **17.96 s against a 5 s
budget**. Reading this trace is how that overrun was found ([ADR-030](Docs/decisions.md)).*

## What does not work yet — measured, not hidden

- **Guardrail latency misses its target (NFR-2).** Input rails take 184 ms. The groundedness rail runs
  one cross-encoder pass per sentence × chunk, and the model already batches, so there is no cheap fix.
  The benchmark had only ever timed the input rails, so an earlier "PASS" measured a subset of the
  requirement ([ADR-029](Docs/decisions.md)).
- **End-to-end latency misses ≤ 20 s (NFR-1):** 34.6 s p50 locally, 50.5 s for a warm query in Docker.
  Tokens stream from the first second, but the groundedness rail runs after the last one.
- **The T3 escalation budget is not enforced.** An over-budget LLM check is flagged, not capped.
- **Citation quality is the weakest area.** 57% of answers cite nothing. In the live demo, the model
  twice cited the wrong source for a correctly quoted fact, and once echoed prompt markup into its
  answer — which *raised* its groundedness score (ADR-030).
- **Plausible-but-unanswerable questions are hedged, not refused** — that is almost all of the 22.7%
  attack-success rate. Refusing instead was measured in Phase 3 at ~20% false refusals.

Each has a costed next step in [`FUTURE.md`](FUTURE.md).

---

## Quickstart

Requires [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/) and a native
[Ollama](https://ollama.com/download).

**One-command demo**

```bash
uv run python scripts/bootstrap_models.py                              # host: pull the 1.9 GB generator into Ollama
docker compose run --rm api python scripts/bootstrap_models.py --warm  # once: download the encoders
docker compose up -d --wait                                            # ~21 s to healthy
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"source":"self"}'
```

Then open **http://localhost:8501**. The first ingest scans every chunk for prompt injection and took
~20 min inside Docker Desktop. Expect a first answer in about a minute; later ones are faster.

**Local development**

```bash
uv python install 3.12 && uv sync
docker compose up -d qdrant
uv run rag ingest --source .
uv run uvicorn rag.api.main:app              # API on :8000, docs at /docs
uv run rag eval retrieval                    # Tier A in seconds, no LLM
uv run pytest -m "not slow and not integration"
```

Every command is listed in [`CLAUDE.md`](CLAUDE.md#commands).

---

## Architecture

```
query → GUARDRAILS in   T0 heuristics · T0 PII · T1 injection · T1 topicality
          ↓
        RETRIEVAL       bge-small dense + Qdrant BM25 → server-side RRF
          ↓
        GENERATION      isolated, untrusted context block · Qwen2.5-3B via Ollama · enforced citations
          ↓
        GUARDRAILS out  T0 PII leak · T2 HHEM groundedness
          └─ score inside the band → T3 LLM check
          ↓
        answer + citations + inspectable GuardrailTrace
```

| Component | Choice | Why |
|---|---|---|
| Generation | Qwen2.5-3B-Instruct Q4_K_M (Ollama) | 7B drops to ~5 tok/s on this CPU |
| Retrieval | bge-small + Qdrant BM25, server-side RRF | One round trip; no second index to keep in sync |
| Rerank | ms-marco-MiniLM-L-6-v2 — **off by default** | Measured lift was −0.106 NDCG@5 at 2.3 s/query (ADR-003) |
| Groundedness | HHEM-2.1-Open, pinned revision | Deterministic; one model serves the rail and the metric; remote code pinned (ADR-028) |
| BERTScore | In-house, distilbert layer 5 | Matches `bert-score` to 2.4e-4 without its 11 dependencies (ADR-024) |
| Tier C judge | Ragas, as an opt-in extra | Its 38 packages never reach the default install or CI (ADR-027) |
| CI gate | Tier A in CI, Tier B local | Tier B generates every answer — ~100 min per run (ADR-026) |
| Demo | Compose + host Ollama | Healthy in 21.4 s, with no 1.9 GB model pull inside Docker (ADR-028) |

## Documentation

- **[Docs/decisions.md](Docs/decisions.md)** — 30 ADRs, including every rejected alternative and each
  time a confident assumption lost to a measurement
- **[Docs/prd.md](Docs/prd.md)** — requirements, constraints, data contracts, NFRs
- **[Docs/plans/](Docs/plans/)** — phased implementation plans, with recorded deviations
- **[FUTURE.md](FUTURE.md)** — what is deferred, and what each fix would cost

## License

MIT — see [LICENSE](LICENSE).

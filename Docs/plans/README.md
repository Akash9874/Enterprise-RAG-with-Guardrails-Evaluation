# Enterprise RAG — Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement these plans task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only, zero-cost RAG service that answers questions about its own codebase
with verifiable citations, enforced by a tiered guardrail pipeline and measured by a
deterministic evaluation harness.

**Architecture:** Hybrid retrieval (dense `bge-small` + Qdrant-native BM25, fused server-side by
RRF) feeds a cross-encoder reranker, then a local Qwen2.5-3B generator with enforced citations.
Guardrails wrap the request in cost-ordered tiers — microsecond deterministic checks, millisecond
encoder classifiers, and an LLM self-check reachable only from one rail's escalation band. The
evaluation harness is built alongside retrieval, not after it, so every tuning decision is
backed by a number.

**Tech Stack:** Python 3.12 (pinned via `uv`), FastAPI, Pydantic v2, Qdrant, Ollama, sentence-
transformers, tree-sitter, `langchain-text-splitters`, Presidio, HHEM-2.1-Open, Streamlit, pytest.

**Spec:** [`Docs/prd.md`](../prd.md) — read it before executing any task.
**Decisions:** [`Docs/decisions.md`](../decisions.md) — the reasoning behind each technology choice.

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the
spec; do not substitute alternatives without amending the PRD.

- **Python 3.12**, pinned via `uv` and `.python-version`. The system Python is 3.14 — never use it.
- **No GPU.** AMD Ryzen 5 7530U, 6C/12T, integrated Radeon ~0.5 GB. All inference is CPU-bound.
- **Resident model budget ≤ 5 GB**; exceeding **6 GB** is a defect (NFR-4).
- **Zero paid-API calls in any default code path** (NFR-9).
- **Generation model:** `qwen2.5:3b-instruct-q4_K_M` via Ollama. Never 7B+.
- **Embedder:** `BAAI/bge-small-en-v1.5`, 384-dim, cosine.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`. Never `bge-reranker-large` (560M ≈ 2 s/query).
- **Groundedness:** `vectara/hallucination_evaluation_model` (HHEM-2.1-Open).
- **Never introduce:** NeMo Guardrails, LlamaGuard, LlamaIndex, full LangChain, TruLens, OpenTelemetry.
  Each was evaluated and rejected — see [`Docs/decisions.md`](../decisions.md).
- **Package root is `src/rag/`.** Modules communicate only through `src/rag/contracts.py`.
- **Models load lazily through the registry.** Never instantiate a transformer at module scope.
- **All config is layered:** `config/settings.yaml` → env vars (`RAG_` prefix) → CLI flags.
  Never hardcode a threshold, model name, or `k` value in source.
- **Tooling:** `uv` for dependencies (never bare `pip`), `ruff` for lint + format, `mypy` on `src/`,
  `pytest` with `@pytest.mark.slow` on anything that loads a real model.
- **Coverage floor:** ≥ 85% on `guardrails/`, `retrieval/`, `eval/` (NFR-8).
- **Commit after every task.** Conventional commit prefixes (`feat:`, `test:`, `chore:`, `fix:`).

---

## Phases

| Phase | Plan | Status | Exit criteria |
|---|---|---|---|
| **0** | [Foundation + walking skeleton](phase-0-foundation.md) | Ready to execute | A question posted to `/query` returns an LLM answer grounded in a hardcoded document. |
| **1** | [Ingestion + retrieval + Tier A eval](phase-1-retrieval-eval.md) | Ready to execute | `rag eval retrieval` prints Recall@5, NDCG@5, MRR, and reranker lift over ~50 golden queries in < 60 s. |
| **2** | [Generation + citations](phase-2-generation.md) | Ready to execute | Every answer carries valid citations resolving to real chunks; fabricated markers stripped and counted. |
| **3** | Guardrails | Roadmap below | Adversarial suite runs; attack-success and false-refusal rates reported; guardrail overhead p50 ≤ 300 ms. |
| **4** | Eval depth + demo | Roadmap below | CI fails on an injected retrieval regression; `docker compose up` reaches a working demo in ≤ 90 s. |

### Why phases 2–4 are roadmaps, not full task plans

Phase 1 produces the numbers that determine Phase 2–4's content: actual chunk sizes, measured
recall, real reranker lift, and the observed score distributions that set every guardrail
threshold. Writing literal test code for a `t_pass` value today would be inventing a number and
then testing against it.

**Expand each phase by re-running `superpowers:writing-plans` against the PRD when the previous
phase's exit criteria are met.** The requirements are already fixed in the PRD; only the
measured constants are open.

---

## Phase 2 roadmap — Generation + citations

| Task | Deliverable | Spec |
|---|---|---|
| 2.1 | Prompt template with structurally isolated context block, labelled untrusted | FR-G2 |
| 2.2 | Citation marker assignment after final ranking (`[1]` = top chunk) | FR-G3 |
| 2.3 | Citation enforcement — parse markers, strip those not in context, count violations | FR-G4 |
| 2.4 | Ollama generation with `temperature=0` for eval, streaming for the API | FR-G1, FR-G5 |
| 2.5 | Refusal on empty retrieval — never answer from parametric memory | FR-G6 |
| 2.6 | `POST /query` wired end-to-end, replacing the Phase 0 skeleton | FR-A1 |
| 2.7 | `stage_timings` propagated through `Answer` | FR-R7 |

## Phase 3 roadmap — Guardrails

| Task | Deliverable | Spec |
|---|---|---|
| 3.1 | `Rail` protocol, `RailContext`, pipeline orchestrator with cost ordering + short-circuit | FR-GR1, FR-GR2 |
| 3.2 | Policy loader — `config/guardrails.yaml`, thresholds, fail-open/fail-closed | FR-GR3, FR-GR5 |
| 3.3 | T0 input heuristics — regex, denylist, length/encoding | FR-GR1 |
| 3.4 | T0 PII rail (Presidio) — input redaction and output leak re-scan | FR-GR1 |
| 3.5 | T1 injection rail — `deberta-v3-base-prompt-injection-v2` | FR-GR1 |
| 3.6 | T1 topicality rail — cosine to corpus centroid, reusing the retrieval embedder | FR-GR1 |
| 3.7 | T2 groundedness rail — HHEM per answer-sentence vs cited chunks | FR-GR1 |
| 3.8 | T3 escalation — LLM self-check, output groundedness only, budget-capped | FR-GR4, FR-GR7 |
| 3.9 | `GuardrailTrace` assembly and `include_trace` on the API | FR-GR6 |
| 3.10 | Ingest-time injection scan + `quarantined` flag | FR-I7 |
| 3.11 | Adversarial suite (~25 cases) with benign controls | FR-E5 |
| 3.12 | `rag bench` asserting guardrail overhead p50 ≤ 300 ms, escalation ≤ 10% | NFR-2, NFR-3 |

## Phase 4 roadmap — Eval depth + demo

| Task | Deliverable | Spec |
|---|---|---|
| 4.1 | Tier B metrics — groundedness, citation precision/recall, BERTScore, refusal correctness | FR-E2 |
| 4.2 | Tier C — Ragas faithfulness + answer relevancy, pluggable judge, opt-in | FR-E3 |
| 4.3 | Provenance block on every report — judge, corpus SHA, model versions, config hash | FR-E4 |
| 4.4 | HTML report with per-query drill-down and baseline diff | FR-E6 |
| 4.5 | `rag eval promote-baseline` — explicit promotion only | FR-E8 |
| 4.6 | CI regression gate — fail on Recall@5 −2 pts or groundedness −3 | FR-E7 |
| 4.7 | Streamlit demo UI with citation panels and the guardrail trace waterfall | FR-U1–U4 |
| 4.8 | `POST /ingest`, `GET /corpus/stats` | FR-A3, FR-A4 |
| 4.9 | README with measured numbers and rejected alternatives | Success criterion 7 |

---

## Execution

Work one task at a time. Each task ends with a commit and is independently reviewable.

```bash
uv run pytest -m "not slow"      # fast loop — run on every change
uv run pytest                    # full suite including model loading
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/
```

**Do not skip the "run the test and watch it fail" step.** A test that has never failed has not
been shown to test anything.

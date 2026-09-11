# Architecture Decision Record

Design decisions for this project, with the evidence behind each. Several reverse the tech
stack originally sketched for this project — those reversals are recorded here rather than
quietly dropped, because the reasoning is the point.

Dated 2026-09-11 unless noted. Status: all **Accepted** at design time; revisit any of them
with a measurement, not an opinion.

---

## ADR-001 — Target CPU-only local inference at zero marginal cost

**Context.** Hardware profiling of the target machine returned: AMD Ryzen 5 7530U (6C/12T,
Zen 3), 15.4 GB RAM, integrated Radeon graphics with ~0.5 GB shared VRAM, 205 GB free disk.
The requirement was that everything be free to run.

**Decision.** All inference runs on CPU. No paid API in any default code path. The resident
model budget is capped at ~5 GB, with 6 GB as a hard defect threshold.

**Consequences.** This single constraint determines nearly every decision below. It rules out
7B+ generation models, 560M rerankers, and any guardrail design that calls an LLM per rail. It
is also the most interesting thing about the project: the constraint forces engineering that a
GPU would have let us skip.

**Rejected.** ROCm on the integrated Radeon — not viable on this iGPU class on Windows.

---

## ADR-002 — Qwen2.5-3B-Instruct Q4_K_M for generation

**Decision.** `qwen2.5:3b-instruct-q4_K_M` via Ollama, ~2.0 GB resident.

**Rationale.** Qwen2.5 is the strongest small model on technical and code content, which is the
corpus. At 3B/Q4 it sustains roughly 12–18 tok/s on 6 Zen 3 cores; a 7B drops to about 5 tok/s,
which makes the demo feel broken regardless of answer quality.

**Consequences.** Token streaming is mandatory, not optional — first token in 1–2 s is what
makes a 20 s total latency tolerable.

**Alternative retained as fallback.** Llama-3.2-3B-Instruct, a close second, configured as a
one-line swap.

---

## ADR-003 — `ms-marco-MiniLM-L-6-v2` for reranking, not `bge-reranker-large`

**Context.** The original stack specified `BAAI/bge-reranker-large` (560M parameters).

**Decision.** Use `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) instead.

**Rationale.** On this CPU, the 560M cross-encoder costs roughly 2 seconds per query to score
30 candidate pairs. MiniLM was expected to do the same work in ~25 ms — about 1/80th the cost —
and to retain most of the ranking quality.

**Consequences.** The quality gap is real but was unmeasured at design time. The eval harness
computes **reranker lift** (ΔNDCG@k with reranking on vs off), so this trade is reported as a
number rather than assumed.

### Measured, 2026-09-11 — the assumption did not survive contact with the harness

First Tier A run against the real corpus (476 chunks, 71 files; 28 hand-authored queries;
file-level ground truth; corpus `da28d90`):

| k_fuse | NDCG@5 (rerank on) | reranker lift | Recall@5 | wall time, 28 queries |
|---|---|---|---|---|
| 30 | 0.646 | **−0.1056** | 0.696 | 76.6 s |
| 20 | 0.674 | −0.0780 | 0.732 | 49.3 s |
| 10 | 0.706 | −0.0460 | 0.750 | 25.3 s |
| 5  | 0.732 | −0.0058 | 0.786 | 13.8 s |
| **off** | **0.752** | — | **0.786** | **1.3 s** |

Two findings, both against the design assumption:

1. **The ~25 ms estimate was wrong by two orders of magnitude.** It was taken on short strings.
   Real chunks here are ~1300 chars at the median and 4355 at the max, and cross-encoder cost
   scales with sequence length: scoring 28 real candidates costs **2.3 s** with all 6 cores
   busy (6.0 s single-threaded). That alone breaches NFR-7 (retrieval p95 ≤ 800 ms).
2. **Lift is negative at every `k_fuse`**, trending to zero only as the reranker is given less
   to do. Fusion-only scores best on every metric: NDCG@5 0.752, Recall@5 0.786, Hit@5 0.857,
   MRR 0.671 — for the whole set in 1.3 s.

**Decision (revised).** `rerank_enabled` now defaults to **false**. The reranker as configured
costs ~75 s per eval run and makes every measured metric worse; keeping it on by default could
not be defended with a number.

**Caveat, stated deliberately.** This ground truth is **file-level**, not chunk-level. A
cross-encoder reorders passages within the candidate set, and file-level scoring may under-credit
that. Chunk-level ground truth could change the quality half of this result — it cannot change
the latency half. Revisit when `relevant_chunk_ids` is populated.

The code and the per-request toggle stay. Lift remains measurable, which is the point: this
decision is reversible the moment a measurement justifies reversing it.

---

## ADR-004 — Build a custom tiered guardrail pipeline; reject NeMo Guardrails and LlamaGuard

**Context.** The original stack specified NeMo Guardrails and/or LlamaGuard. Four guardrail
families were requested: groundedness, prompt injection, PII, and topical policy.

**Decision.** Build an explicit tiered pipeline. Rails are ordered by cost — T0 deterministic
(microseconds), T1 small encoder classifiers (~10–40 ms), T2 NLI cross-encoder (~150 ms), T3
LLM self-check reachable **only** from the output groundedness rail's escalation band.

**Rationale.**

- **LlamaGuard is 8B.** It alone exceeds the entire 5 GB model budget. There is no version of
  this project where it fits.
- **NeMo Guardrails' rails are LLM-call-based.** The naive composition — injection check,
  generate, groundedness check, policy check — is four sequential LLM calls. At this hardware's
  throughput that is 60–90 seconds per query. Dead on arrival.
- The tiered design gets the common case to **one LLM call plus ~260 ms of rails**, with
  escalation firing on a minority of requests.

**Consequences.** More code to own and test than adopting a framework. In exchange, the control
flow supports early exits, escalation bands, per-stage timing, and a full decision trace — none
of which fits cleanly into a declarative rail framework. For a portfolio project the custom
pipeline is also the stronger signal: it demonstrates the mechanism rather than the wrapper.

**Revisit if.** The project ever moves to hardware where per-rail LLM calls are cheap.

---

## ADR-005 — HHEM-2.1-Open for groundedness, used as both guardrail and eval metric

**Decision.** `vectara/hallucination_evaluation_model` (184M, Apache-2.0) scores factual
consistency of each answer sentence against the chunks that sentence cites.

**Rationale.** Groundedness is normally checked by asking an LLM, which costs seconds and is
non-deterministic. HHEM is purpose-built for exactly this premise/hypothesis judgement, runs in
~150 ms on CPU, and is deterministic — which means it can gate CI.

**Consequences.** One model serves two roles: the T2 output rail and the Tier B eval metric.
That reuse keeps the RAM budget intact and makes the guardrail and the metric definitionally
consistent — the thing being measured is the thing being enforced.

**Risk.** HHEM may disagree with human judgement. Mitigation: hand-check 20 cases and report
the agreement rate in the README. Treat it as a strong signal, not an oracle.

---

## ADR-006 — Use `langchain-text-splitters` only; reject LangChain and LlamaIndex as orchestrators

**Context.** The original stack specified LlamaIndex or LangChain. Dependency trees were
resolved with `uv pip compile` against Python 3.12 to quantify the cost:

| Option | Total packages | New vs hand-rolled core |
|---|---|---|
| Hand-rolled core (fastapi, qdrant-client, sentence-transformers, tree-sitter, ollama) | 60 | — |
| `+ langchain-text-splitters` | 80 | **+20** (~20 MB) |
| `+ full LangChain` (langchain, -community, -qdrant, -huggingface) | 103 | **+43** (~80 MB) |

`torch` and `transformers` appear in **all** options via sentence-transformers, so they are not
a differentiator — at ~250 MB they dominate disk either way.

**Decision.** Take `langchain-text-splitters` for code and markdown chunking. Hand-roll
retrieval, generation, guardrails, and evaluation.

**Rationale.** The full tree adds `sqlalchemy`, `aiohttp`, `langgraph` ×4, and
`langchain-community`, costing ~200 MB of import-time RAM and several seconds of cold import on
every test run and reload. The decisive cost is not resources though — it is that LCEL's
dataflow model fights the guardrail control flow, which needs early exits, escalation bands, and
per-stage latency attribution. Implementing that inside custom `Runnable`s is hand-rolling with
extra steps and worse observability.

Meanwhile `RecursiveCharacterTextSplitter` (language-aware separators for code) and
`MarkdownHeaderTextSplitter` (header-path-preserving) are well-tested solutions to problems not
worth re-solving.

---

## ADR-007 — Qdrant with server-side RRF fusion

**Decision.** Qdrant as the single vector store, holding both dense and native sparse (BM25)
vectors, with fusion performed server-side via the Query API's `prefetch` +
`FusionQuery(Fusion.RRF)`.

**Rationale.** One round trip, one index, no client-side state. RRF requires no score
normalisation between retrievers, which matters because cosine similarity and BM25 are not on
comparable scales — any weighted-sum alternative would need re-tuning every time the embedder
changed.

**Rejected.**

- **pgvector** — requires running Postgres, and its full-text search is a weaker BM25 than
  Qdrant's native sparse vectors. Heavier for strictly less capability here.
- **Client-side `rank_bm25`** — a second index to keep in sync, in-process memory against a
  5 GB budget, and a full rebuild on every restart.

---

## ADR-008 — Deterministic evaluation backbone; LLM judge is an optional tier

**Context.** Ragas and TruLens both make every metric depend on a judge LLM. Under ADR-001 the
judge would be a local 3B model, which correlates poorly with human judgement.

**Decision.** A three-tier ladder. **Tier A** (retrieval: Recall@k, Precision@k, MRR, NDCG@k,
reranker lift) and **Tier B** (generation: HHEM groundedness, citation precision/recall,
BERTScore, refusal correctness) require **no LLM at all** and gate CI on every commit. **Tier C**
(Ragas faithfulness and answer relevancy) is opt-in, with a pluggable judge.

**Consequences.** The numbers this project reports as its backbone are deterministic,
reproducible, and fast. Every report stamps the judge model, corpus commit SHA, model versions,
and config hash — a Tier C score without its judge named is a misleading number.

**Rejected.** TruLens — fully overlaps Ragas for this use and adds a dashboard dependency.

---

## ADR-009 — Build the evaluation harness in Phase 1, not at the end

**Decision.** Tier A metrics ship alongside retrieval, before generation, guardrails, or UI.

**Rationale.** Retrieval cannot be tuned without measurement. Chunk size, `k` values, fusion
parameters, and reranker choice are all empirical questions, and answering them by intuition is
the single most common failure in RAG projects. Building the harness first means every
subsequent tuning decision is backed by a number from this project's own tooling.

**Consequences.** Phase 1 is the longest phase and produces no user-visible feature. Accepted
deliberately — this is the structural decision the whole plan rests on.

---

## ADR-010 — Index the project's own repository as the corpus

**Decision.** The system's primary corpus is its own source code and documentation.

**Rationale.** No data licensing concerns; the corpus grows with the project; the golden set can
be authored by the person who wrote the source, so ground truth is genuinely known; and "ask
this system about itself" is an immediately legible demo for a reviewer with ten minutes.

**Consequences.** The golden set must be re-validated when the code changes materially — chunk
IDs are content hashes and will shift. Accepted: `chunk_id` stability is scoped to unchanged
content, and golden entries also record `relevant_files` as a coarser fallback.

---

## ADR-011 — Pin Python 3.12 via `uv`

**Context.** The development machine has Python 3.14.0 installed.

**Decision.** Pin 3.12 through `uv python install 3.12` and `.python-version`.

**Rationale.** 3.14 is ahead of wheel availability for `spacy` (and therefore Presidio) and parts
of the torch ecosystem. Building those from source on this machine is not a reasonable use of
time.

---

## ADR-012 — Cut ONNX quantization, OpenTelemetry, TruLens, and the toxicity rail from scope

**Context.** An earlier eight-phase plan included INT8 ONNX quantization of the encoder models,
distributed tracing, TruLens alongside Ragas, and a toxicity classification rail. The project was
deliberately re-scoped to intermediate-to-advanced.

**Decision.** All four are cut. Recorded in PRD §3.2 as binding non-goals.

**Rationale.**

- **ONNX/INT8** — genuine work with real gains (~3× RAM, ~2× latency on CPU encoders), but it is
  optimising a system that does not exist yet, and the portfolio signal is modest relative to
  the effort. Listed as future work.
- **OpenTelemetry** — the guardrail trace already captures per-stage timings, which is what this
  single-node system actually needs. Structured logs cover the rest.
- **TruLens** — fully overlapping with Ragas (ADR-008).
- **Toxicity rail** — near-zero value on a code-and-docs corpus, at a cost of ~400 MB against a
  5 GB budget. The remaining four rail families carry the feature.

**Consequences.** Target size lands around 2,500–3,500 LOC including tests. Finishing matters
more than ceiling for a portfolio project; scope creep back toward the original plan is listed
as a tracked risk in PRD §12.

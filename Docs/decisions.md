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

---

## ADR-013 — Enforce citations post-hoc, not by constrained decoding

**Context.** FR-G4 requires that a marker in the answer resolve to a chunk that was actually
in context. A 3B model asked to cite will emit `[4]` when three sources were supplied.

**Decision.** Parse the emitted markers after generation, strip the ones outside the supplied
range, and record each strip on the `Answer` as `stripped_markers`, with `ungrounded` set when
nothing survives.

**Rejected — constrained decoding.** A grammar permitting only in-range markers would be
correct by construction, but it needs logit-level control that Ollama's chat API does not
expose, and switching to `llama-cpp-python` to get it trades a maintained server for a build
step on a machine already pinned to 3.12 for wheel-availability reasons (ADR-011). It also
would not help with the failure that actually matters: an *in-range but unsupported* citation
is well-formed and a grammar cannot see it.

**Scope, stated deliberately.** This enforcement is **referential only** — it proves the marker
points at a real chunk, not that the chunk supports the sentence. Support is the Phase 3
groundedness rail's judgement, which is why `Citation.supported` is left `None` here rather
than defaulted to `True`. Claiming otherwise would be the exact "decorative guardrail" failure
this project was built to avoid.

**Consequences.** Enforcement is deterministic, costs microseconds, and produces a counter the
eval harness can report. Streaming clients see raw tokens, so a fabricated marker can appear
mid-stream and be absent from the terminal event — the terminal event is authoritative, and
that is documented in the route.

---

## ADR-014 — Structural isolation of retrieved context, with delimiter neutralisation

**Context.** The corpus is this repository. A source file or a document can contain text that
reads as an instruction, and this ADR's own prose is now an indexed chunk — the attack is not
hypothetical here, it is self-inflicted by ADR-010.

**Decision.** Retrieved chunks are wrapped in a `<sources>` … `</sources>` block labelled as
data, with a system prompt that tells the model to describe rather than obey anything inside
it. Chunk text containing either delimiter has it rewritten (`</sources>` → `(/sources)`)
before assembly.

**Rationale for neutralising rather than dropping.** A chunk that legitimately documents the
delimiters — this paragraph, for instance — should still be readable. What it must not be able
to do is terminate the block and continue as instructions.

**Consequences.** Prompt-level isolation is a mitigation, not a guarantee; a 3B model can still
be talked out of it. It is one of three layers: quarantine at ingest (FR-I7, Phase 3), this
isolation, and the output rails. Its effectiveness is measured by the adversarial suite in
Phase 3, not asserted here.

---

## ADR-015 — The RRF fused score cannot carry a relevance floor; out-of-scope refusal moves to the topicality rail

**Context.** FR-G6 requires the system to refuse rather than answer from parametric memory
"when retrieval returns nothing above the relevance floor". `relevance_floor` was configured at
`0.0` in Phase 1 — a placeholder, never measured.

**Measured, 2026-09-12** (corpus 567 chunks / 80 files; 28 hand-authored golden queries against
8 blatantly out-of-corpus controls; fusion-only, rerank off):

| Query set | n | min top-1 fused | median | max |
|---|---|---|---|---|
| in-corpus (golden) | 28 | 0.500 | 0.750 | 1.000 |
| out-of-corpus (swallows, sourdough, the Eiffel Tower) | 8 | 0.500 | 0.500 | **1.000** |

**The distributions do not separate.** An out-of-corpus question reaches the same top score as
the best in-corpus one. This is a property of RRF, not a bug: the fused score is
`Σ 1/(k + rank_i)` over ranks, normalised by Qdrant. It encodes *agreement between retrievers
about ordering*, not *similarity to the query*. Both retrievers will happily rank something
first when nothing is relevant, and that unanimity scores exactly as high as a real hit.

**Decision.** No floor value on the fused score can implement FR-G6's intent, so none is
invented. `relevance_floor` stays at `0.0` and is documented as guarding only the genuinely
empty case — filters excluding everything, or an un-ingested corpus — which it does correctly
and which the tests cover. **Out-of-scope refusal is reassigned to the T1 topicality rail**
(FR-GR1, Phase 3): cosine distance from the query embedding to the corpus centroid *is* a
calibrated similarity, and it is the right instrument for this.

**Consequences, stated plainly.** Until Phase 3 lands, this system will answer an out-of-scope
question using whatever the corpus ranked first, rather than refusing. That is a real gap
against FR-G6 and it is recorded here rather than papered over with a floor that looks like a
threshold and separates nothing.

**Rejected — floor on the dense cosine score instead.** Workable in principle, but it needs
retrieval to surface the pre-fusion dense score, which server-side RRF deliberately does not
return (ADR-007), and it would duplicate the topicality rail one phase early.

---

## ADR-016 — Citation prompt wording, chosen by measurement after the obvious improvement lost

**Context.** FR-G4 requires every factual sentence to carry a citation marker. Enforcement can
only remove markers that do not resolve; it cannot make the model emit one. Whether answers get
cited at all is therefore a property of the prompt, and it is measurable.

**Measured, 2026-09-12** — 28 hand-authored golden queries, corpus of 567 chunks / 80 files,
`temperature=0`, fusion-only retrieval, `k_final=5`:

| Prompt variant | answers citing nothing | mean citations / answer | p50 latency |
|---|---|---|---|
| **A — citation rule stated first, plainly** | **12/28 (43%)** | **0.61** | 34.6 s |
| B — `CITATION FORMAT` heading, worked example, rule moved last for recency | 18/28 (64%) | 0.39 | 34.2 s |

**Decision.** Keep variant A. Variant B was the change expected to help — a concrete worked
example and a recency-favoured position are both standard advice for small instruct models — and
it made the metric **21 points worse**. It was reverted.

**Why this is recorded rather than quietly dropped.** This is the second time in this project
that a confident prior lost to a measurement (ADR-003 was the first). The value is not the
wording; it is that a prompt change is now a thing with a number attached. The comment above
`SYSTEM_PROMPT` says so at the call site, because the next person to "improve" that string will
not read this file first.

**Not claimed.** No mechanism is offered for *why* B lost. With n=28 and one model the honest
statement is that it lost, not why. A larger golden set would be needed to say more, and
inventing an explanation would be exactly the intuition this table exists to displace.

**Open gap, stated plainly.** 43% of answers still carry no citation at all. Enforcement is
working as specified — it strips fabricated markers, and over these 28 queries there were
**zero** to strip — but a well-formed uncited answer passes straight through as `ungrounded`.
Closing that is Phase 3 and 4 work: the T2 groundedness rail acts on the flag, and Tier B's
citation-recall metric (FR-E2) turns it into a gated number. Phase 2 flags it honestly and
does not pretend to solve it.

---

## ADR-017 — Pin `transformers < 5` to keep HHEM-2.1-Open working

**Context.** ADR-005 chose HHEM as the groundedness model, serving as both the T2 rail and the
Tier B eval metric. On first load under the resolved stack it failed outright:

```
AttributeError: 'HHEMv2ForSequenceClassification' object has no attribute
'all_tied_weights_keys'
```

HHEM ships its own modelling code via `trust_remote_code`, written against the transformers 4.x
API. `uv` had resolved transformers **5.17.0**, whose internals it does not match. The load also
reported `t5.transformer.encoder.embed_tokens.weight MISSING`, so the model was not merely noisy
— it was not loading its weights.

**Decision.** Pin `transformers < 5`. The resolver settles on transformers 4.57.6 with
sentence-transformers 5.7.0 and torch 2.14.0.

**Verified, not assumed.** HHEM was re-run against the five premise/hypothesis pairs published on
Vectara's model card, and all five land on the expected side:

| premise → hypothesis | score | expected |
|---|---|---|
| capital of France is Berlin → is Paris | 0.011 | low |
| in California → in United States | 0.647 | high |
| in United States → in California | 0.129 | low |
| person on a horse jumps… → person outdoors on a horse | 0.897 | high |
| boy on skateboard on a bridge → skates down the sidewalk | 0.185 | low |

The full test suite and `rag eval retrieval` were re-run after the downgrade; nothing regressed.

**Rejected — swap HHEM for a generic NLI cross-encoder** (`nli-deberta-v3-base` or
`DeBERTa-v3-base-mnli-fever-anli`, both the same size class, both plain
`AutoModelForSequenceClassification` with no remote code and no version pin). It would have
avoided the pin, but it discards the property ADR-005 selected HHEM *for*: it is purpose-built
for exactly this premise/hypothesis judgement, and it doubles as the eval metric so that the
thing enforced and the thing measured are definitionally the same. A version pin is a cheaper
price than losing that.

**Consequence.** The project is held one major version back on transformers until Vectara
updates the remote code. Recorded as a known constraint rather than discovered later by whoever
next runs `uv lock --upgrade`.

---

## ADR-018 — The Phase 3 model budget, measured; PRD §4.1 was optimistic

**Context.** PRD §4.1 budgeted the guardrail models from published parameter counts, before any
of them had been run on this machine.

**Measured 2026-09-12** on the reference CPU. Loaded in sequence into one process, so the
increments share a single torch runtime — which is why they are each smaller than the same model
measured alone:

| Step | resident | increment | per call |
|---|---|---|---|
| baseline + imports | 0.04 GB | — | — |
| + `bge-small` embedder | 0.51 GB | +0.47 | — |
| + Presidio (`en_core_web_sm`) | 0.60 GB | +0.09 | 7 ms |
| + deberta injection | 1.04 GB | +0.45 | 120 ms warm, 306 ms cold |
| + HHEM groundedness | **1.42 GB** | +0.38 | 197 ms warm, 4.2 s cold |

**NFR-4 passes with room to spare**: 1.42 GB in-process, plus Ollama (~2.0 GB, separate process)
and Qdrant (~0.3 GB, container) for roughly **3.7 GB against a 6 GB ceiling**.

**Two estimates in PRD §4.1 were wrong, in opposite directions.**

1. **Presidio would have blown its 0.30 GB line by 3.6×.** The default `AnalyzerEngine()`
   silently downloads and loads `en_core_web_lg` — 382 MB on disk, **1.09 GB resident, 102 ms
   per call**. Pinning `en_core_web_sm` *and* scoping the entity list to the five types the
   policy names brings that to **0.48 GB and 7 ms** — a 15× latency reduction. Both are
   load-bearing and both are asserted in tests, because the expensive path is the default one
   and nothing about it looks wrong.
2. **The injection classifier is 3× its latency budget.** PRD §7.4 estimated ~40 ms; it measures
   120 ms warm. The estimate appears to have been taken from parameter count rather than a run.

**Consequence.** NFR-2 (guardrail overhead p50 ≤ 300 ms) still passes at **184 ms p50, 220 ms
p95** across the 28 golden queries, because the T0 rails cost microseconds and short-circuit a
large share of hostile traffic before the classifier runs. The tiering is not a nicety here — it
is the only reason a 120 ms classifier fits inside a 300 ms budget at all.

**PRD §4.1 is updated to the measured figures rather than left as the original estimates.**

---

## ADR-019 — Topicality thresholds, measured; this is what ADR-015 deferred here

**Context.** ADR-015 established that the RRF fused score cannot detect an out-of-scope question
— in-corpus and out-of-corpus scores overlap completely — and reassigned the job to this rail.
That left a promise to keep: show that cosine-to-centroid actually does better.

**Measured 2026-09-12** — 28 hand-authored golden queries against 12 blatantly out-of-corpus
controls, cosine similarity to the mean of all 567 indexed dense vectors:

| | n | min | median | max |
|---|---|---|---|---|
| in-corpus | 28 | 0.564 | 0.687 | 0.779 |
| out-of-corpus | 12 | 0.422 | 0.497 | 0.644 |

**It separates far better than RRF, but not perfectly.** The lowest in-corpus query (0.564) sits
below the highest out-of-corpus one (0.644), so there is no threshold that is simultaneously
perfect in both directions. That overlap is the honest finding; the thresholds are a chosen
trade, not a clean cut.

Threshold sweep:

| t_block | in-corpus wrongly refused | out-of-corpus refused |
|---|---|---|
| 0.50 | 0/28 | 7/12 (58%) |
| 0.52 | 0/28 | 9/12 (75%) |
| **0.55** | **0/28** | **10/12 (83%)** |
| 0.57 | 1/28 (3.6%) | 11/12 (92%) |

| t_pass | in-corpus passing cleanly | out-of-corpus reaching "pass" |
|---|---|---|
| 0.60 | 27/28 | 1/12 |
| **0.65** | **25/28 (89%)** | **0/12** |
| 0.70 | 12/28 (43%) | 0/12 |

**Decision.** `t_block = 0.55`, `t_pass = 0.65`. 0.55 is the last point before false refusals
begin — 0.57 buys one extra catch at the cost of refusing a real question, and on a system whose
false-refusal rate is a headline metric that is the wrong side of the trade. 0.65 is the lowest
value at which nothing out-of-scope reaches a clean pass.

**Consequence.** The five queries landing between the thresholds hedge rather than resolve. That
is the band doing its job. Note this rail does **not** escalate to T3 — FR-GR4 reserves that for
output groundedness — so a hedge here is final, and costs no latency.

**Against ADR-015's gap:** out-of-scope refusal now works, measured at 5/6 in the adversarial
suite. The gap that ADR recorded is closed, and the residual 1/6 is reported rather than hidden.

---

## ADR-020 — PII threshold, and the rail-interaction bug the adversarial suite found

**Context.** The first adversarial run scored the PII family at 50% attack success. Both failures
had causes worth recording, and one of them was a genuine design error.

### 1. `t_block = 0.5` silently missed every phone number

Presidio's confidence is per-recogniser, not a uniform scale. Measured on this machine:

| input | entity | score |
|---|---|---|
| `bob.smith@example.com` | EMAIL_ADDRESS | 1.00 |
| `4111111111111111` | CREDIT_CARD | 1.00 |
| `+1-415-555-0132` | PHONE_NUMBER | **0.40** |
| `SSN 123-45-6789` | *(nothing detected)* | — |

0.40 is Presidio's "pattern matched, not independently validated" level, and the guessed 0.50 sat
just above it. Lowering to **0.40** costs nothing measurable: across the 28 real golden queries,
**0/28 produce any detection at any threshold down to 0.30**, so no false-positive budget is
being spent.

**Not fixed, and stated plainly:** `US_SSN` is in the policy's entity list but Presidio's
recogniser did not fire on a bare `123-45-6789` — it wants surrounding context. The entity stays
configured, but this rail must not be described as covering SSNs on the strength of the config.

### 2. Redaction output tripped the next rail — rails were not independent

The second failure was the interesting one. The PII rail redacts before the injection classifier
runs, and the classifier then scored the *redacted* text:

| text handed to the injection classifier | injection probability |
|---|---|
| `My email is bob.smith@example.com - which file handles ingestion?` | **0.0007** |
| `My email is <EMAIL_ADDRESS> - which file handles ingestion?` | **0.9350** |

**Redacting a benign query manufactured an attack out of nothing.** A user who mentions their own
email while asking a real question got blocked — and the benign controls, all PII-free, could not
see it.

Placeholder-shopping was tried and abandoned: `[EMAIL_ADDRESS]` scores 0.171, `{EMAIL_ADDRESS}`
0.359, `[redacted EMAIL_ADDRESS]` 0.996 and `EMAIL_ADDRESS_REDACTED` 1.000, while
`<PHONE_NUMBER>` in the same sentence position scores 0.0003. The behaviour is erratic enough
that no placeholder can be called safe on this evidence.

**Decision — the fix is architectural, not cosmetic.** The pipeline no longer feeds one rail's
redaction to the next. A redaction changes what the *caller* receives; every rail inspects the
original text. This is what the rail contract already required — rule 5, "no rail may depend on
another rail having run" — so the bug was a violation of the existing design rather than a gap in
it. Chaining rail outputs makes each rail's input a function of every rail before it, which is
exactly how a benign query becomes an attack.

**Suite change.** Two PII-bearing *benign* controls were added, because the original controls
were all PII-free and structurally could not catch this. A suite that cannot see a bug is as much
the finding as the bug.

**Result.** PII attack success 50% → **0%**, with false refusals still 0/12.

---

## ADR-021 — Groundedness must score per chunk and take the max; HHEM's 512-token window

**Context.** With groundedness scoring each sentence against the concatenated text of the chunks
it cited — the definition in `eval/CLAUDE.md` — the rail produced **no usable signal at all**:

| | scores |
|---|---|
| answerable questions (n=10) | 0.000 0.000 0.000 0.000 0.002 0.138 0.140 0.184 0.343 0.482 |
| unanswerable questions (n=4) | 0.000 0.000 0.000 0.000 |

A correct, cited answer and a confabulated answer about a non-existent Redis cache both scored
0.000. No threshold separates those distributions, and a rail configured on them would have been
decorative — the exact failure this project was built to avoid.

**Two causes, both measured.**

1. **Silent truncation.** HHEM's window is 512 tokens; a concatenated premise reached 1650 and
   transformers warned it would be truncated. The support for the claim was in the part cut off.
   The same sentence scored **0.184** against a concatenated premise and **0.969** scored against
   chunks individually.
2. **The cited chunk is the wrong premise for this question.** Phase 2 measured that 43% of
   answers carry no citation at all (ADR-016); under the cited-chunk definition every one of
   those scores 0 regardless of whether it was actually grounded.

**Decision.** Score each sentence against each retrieved chunk separately and take the **max** —
nothing is concatenated, so nothing is truncated, and a claim counts as grounded if *any*
retrieved passage supports it.

| | scores under per-chunk max |
|---|---|
| answerable | 0.075 0.109 0.437 0.509 0.527 0.616 0.876 0.900 0.940 0.952 |
| unanswerable | 0.077 0.127 0.148 0.455 |

Still overlapping — this is a hedging signal, not an oracle — but there is now a real difference
between the distributions where before there was none.

**Deliberate divergence from `eval/CLAUDE.md`.** That file defines groundedness against *cited*
chunks. The rail now asks a narrower question — *is this answer hallucinated?* — for which any
retrieved passage is valid evidence. Whether the answer cited the *right* chunk is citation
precision, a different metric, and conflating the two is what destroyed the signal. Tier B keeps
the cited-chunk definition; the rail and the metric now differ on purpose, and that difference is
recorded here rather than left for someone to trip over.

**Also measured: strip citation markers before scoring.** Leaving `[1]` in the hypothesis costs
~0.10–0.12 on every supported sentence (0.944→0.846, 0.970→0.849, 0.965→0.869) while leaving a
contradicted one unchanged (0.507→0.511). The marker is not part of the claim, and keeping it
compresses precisely the separation the rail depends on.

---

## ADR-022 — A rail's trip verdict is declared in policy, never hardcoded

**Context.** The first full-path adversarial run — input rails, retrieval, generation, output
rails — returned a **91.7% false refusal rate: 11 of 12 benign controls refused**, every one by
the groundedness rail. Attack success would have looked excellent. The suite could only say so
because it has benign controls.

**Cause.** The groundedness rail returned a hardcoded `refuse` when the score fell below
`t_block`, while `config/guardrails.yaml` declared `action: hedge` for it. The policy field was
being ignored, so FR-GR3 ("policy is declarative … per-rail `action`") was not actually
implemented, and FR-GR5 ("quality rails fail open with a hedge") was contradicted in code.

The two defects compounded: badly calibrated thresholds (ADR-021) pushed most real answers below
`t_block`, and the hardcoded verdict turned every one of those into a denial rather than a
warning.

**Decision.** Every rail's trip verdict comes from `policy.trip_verdict`. `allow` maps to `pass`,
so a policy can neutralise a rail without disabling it — the rail still runs and still reports its
score to the trace, which is the difference between "we decided this is fine" and "we stopped
looking".

**Consequence.** Groundedness now hedges: the answer is returned with a warning that it may not be
fully supported, rather than withheld. Refusing remains available to anyone who sets
`action: refuse`, which is the point of having the field.

**The general lesson, recorded because it generalises.** A guardrail's failure mode is not only
"lets an attack through" — it is equally "refuses everything and reports a perfect attack-success
rate". Only the benign half of the suite distinguishes those, and this run is the concrete
demonstration that the benign half earns its place.

---

## ADR-023 — Evaluation fixtures were in the corpus; `eval/` is never indexed

**Context.** The loader indexes every `.yaml`, and only `eval/reports` was gitignored. The golden
set and the adversarial suite were therefore retrievable context. A golden query could retrieve
the file containing its own question text, and adversarial payloads sat in the index beside real
code.

**Decision.** `eval/` is a hard loader exclusion (`ALWAYS_SKIP_PREFIXES`), asserted by a test.

**Measured effect — a controlled A/B on the same tree** (2026-09-13, fusion only, 28 hand
queries, 1,047 chunks). The only difference between the rows is the two `eval/` chunks:

| | NDCG@5 | MRR | Recall@5 | Hit@5 |
|---|---|---|---|---|
| `eval/` indexed | 0.650 | 0.607 | 0.661 | 0.714 |
| `eval/` excluded | **0.684** | **0.631** | 0.661 | 0.714 |

The contamination cost **−0.034 NDCG@5 and −0.024 MRR** and left recall and hit rate unchanged.
The golden file did not knock relevant chunks out of the top 5; it outranked them. That damage
is visible only in rank-sensitive metrics, which is why Recall@5 alone never showed it.

**The earlier Tier A figures are not comparable, and this ADR does not pretend they are.** The
Phase 1–3 numbers (NDCG@5 0.716, Recall@5 0.750 over 567 chunks) were measured on a corpus that
has since nearly doubled with Phase 2–4 code and documentation. Against today's corpus, Recall@5
is 0.661. That drop comes from corpus growth, not from this change, and it is reported rather
than explained away.

**Found on the way: the ingest scan now quarantines 29 chunks, and not only adversarial tests.**
Most are unit tests that contain attack strings deliberately (`test_prompt.py`,
`test_guardrail_pipeline.py`, `test_rail_injection.py` …). The classifier also quarantined
product code: `src/rag/cli_eval.py` (2 chunks), `src/rag/guardrails/pipeline.py`,
`src/rag/guardrails/policy.py`, and this phase's plan document. Those chunks are now invisible to
retrieval. Two consequences:

1. CI cannot skip the scan (`--no-scan`) without indexing a different corpus from the one the
   product serves. The baseline must come from a scanned index.
2. The injection threshold (`t_block 0.80`, still UNMEASURED) is quarantining the guardrail code
   that *describes* injection. That is a false-positive rate on real code, recorded here, not
   tuned in this phase.

**Also.** The golden file is renamed `golden.yaml` and hashed with line endings normalised, so a
CRLF checkout and the LF CI runner agree. Stale chunk references now fail the run (exit 2)
instead of scoring as misses.

---

## ADR-024 — BERTScore in-house on distilbert; Tier B groundedness per cited chunk

**Context.** FR-E2 asks for BERTScore against golden answers. The `bert-score` package adds
**11 dependencies** (matplotlib, pandas, …) to the resolved set for what is a greedy cosine match
over transformer hidden states. Its recommended model, `microsoft/deberta-xlarge-mnli`, is a
**3,036 MB** download (measured from the Hugging Face API, 2026-09-13).

**Decision.** Implement the metric in `rag.eval.metrics.generation.bertscore_f1` (numpy, ~15
lines) over `distilbert/distilbert-base-uncased` hidden layer 5 — the model and layer
`bert-score` itself uses by default for that checkpoint. 268 MB download, **+0.14 GB resident**
in the eval process (measured). The API process never loads it.

**Parity, measured** in a throwaway `uv run --with bert-score` environment, five pairs,
`idf=False`, no baseline rescaling:

| pair | ours | bert-score | \|Δ\| |
|---|---|---|---|
| RRF paraphrase | 0.85318 | 0.85318 | 0.000000 |
| idempotency paraphrase | 0.90731 | 0.90755 | 0.000243 |
| reranker paraphrase | 0.82685 | 0.82685 | 0.000000 |
| unrelated | 0.69862 | 0.69862 | 0.000000 |
| identical | 1.00000 | 1.00000 | 0.000000 |

Max |ΔF1| = **0.000243**, inside the 1e-3 tolerance set before running it.

**Rejected.** The `bert-score` package: identical numbers for 11 more packages. The
deberta-xlarge-mnli model: best human correlation in the BERTScore paper, 11× the download and
most of the remaining model budget. **Consequence stated plainly:** distilbert scores are not
comparable to published BERTScore figures that use other models. They are comparable run-to-run,
which is what a regression harness needs.

**Also decided — Tier B metric definitions.**

- *Groundedness* scores each sentence against each **cited** chunk separately and keeps the max.
  This keeps `eval/CLAUDE.md`'s cited-chunk definition while never concatenating premises, which
  silently truncated at HHEM's 512-token window (ADR-021). The *rail* keeps its any-retrieved-chunk
  definition. The two differ on purpose: the rail asks "is this hallucinated?", the metric asks
  "did the citations carry it?".
- *Citation precision* is `None`, not 0, for an answer with no citations; 0/0 is undefined.
- *Citation recall* treats every sentence as a claim. Stated simplification.
- A citation marker placed after the full stop attaches to the following sentence. Known
  limitation of sentence-level scoring.

---

## ADR-025 — The gate checks each provenance half, and "not run" is not "passed"

**Decision.** FR-E7's thresholds — Recall@5 may not drop more than 2 points, mean groundedness not
more than 3 — are applied to the `hand` and `synthetic` halves **separately**, as dotted paths in
`eval.gate_max_drop` (e.g. `tier_a.by_provenance.hand.recall@5`). A pooled figure would let the
easier synthetic half absorb a hand-authored regression — the pooling `eval/CLAUDE.md` forbids.

**Rules, each asserted by a test.**

- A metric present in only one report is `not_run`, not a failure. That is what lets CI gate
  Tier A alone (ADR-026) without pretending Tier B passed.
- A run in which *nothing* was gated **fails**. A gate that checked nothing must not print PASS.
- A different golden-set hash fails outright; comparing across golden sets is invalid.
- A drop exactly at the threshold passes. `0.78 − 0.80` is `−0.020000000000000018` in IEEE-754,
  so the comparison carries a 1e-9 epsilon — without it the boundary would fail by rounding.
- Promotion refuses a report whose corpus commit is `-dirty`: its numbers correspond to no commit.
- Only `rag eval promote-baseline` writes `eval/baselines/`. A test asserts that writing a report
  leaves the baseline file untouched.

**Consequence, stated plainly: the gate is strict, not tolerant.** With 28 hand queries, one
single-file query flipping from hit to miss moves hand Recall@5 by 1/28 = **3.6 points** — more than
the 2-point threshold. So the gate trips on any single-query retrieval loss. That is defensible
only because Tier A is bit-reproducible (`tests/integration/test_determinism.py`); on a noisy
metric this threshold would be flake, not signal. Growing the hand half is the way to make the
threshold mean what FR-E7 intended.

---

## ADR-026 — CI gates Tier A; the groundedness gate runs locally

**Context.** FR-E7 gates on Recall@5 and on mean groundedness. Groundedness needs generated answers:
a 1.9 GB model pull, then a CPU generation per golden query. Measured on the reference laptop, the
first full Tier B run over 34 hand queries took **98.5 minutes** (under some CPU contention). A shared
CI runner has fewer cores than that laptop.

**Decision.** A `retrieval-gate` CI job starts a Qdrant service container, indexes the checkout with
the injection scan on, runs Tier A and gates Recall@5 per provenance half. `rag eval gate` enforces
the groundedness threshold locally before any baseline is promoted. In CI the Tier B checks report
`not_run`, never `pass` (ADR-025).

**Proof the gate fails when it should.** A throwaway branch set `k_dense` and `k_sparse` to 1. Run
[34758239606](https://github.com/Akash9874/Enterprise-RAG-with-Guardrails-Evaluation/actions/runs/34758239606):

| | Recall@5 (hand) | NDCG@5 | MRR | Hit@5 | gate |
|---|---|---|---|---|---|
| baseline (`07b032b`) | 0.661 | 0.683 | 0.643 | 0.750 | — |
| injected regression | **0.464** | 0.398 | 0.411 | 0.536 | **FAIL** (Δ −0.196, max drop 0.020) |

**Laptop and runner agree exactly.** The same commit indexed locally with the same scan gave
identical Tier A to four decimals, with **0 of 28 queries** returning a different top-5 order.
The baseline was promoted from the CI artifact because CI is where the gate runs. On this evidence it
would have been identical promoted from the laptop.

**The finding this ADR most needs to record: the gate's margin is eaten by the corpus itself.**
This system indexes its own repository, so every commit changes the corpus. The first green run
after promotion was on a tree that added Tier C, the API endpoints and the UI, with **no retrieval
code changed**:

| | chunks | quarantined | Recall@5 (hand) |
|---|---|---|---|
| baseline `07b032b` | 1,194 | 35 | 0.661 |
| green run `984d316` | 1,275 | 37 | 0.643 (Δ **−0.018**, gate PASS) |

Corpus growth alone used 1.8 of the 2.0 points FR-E7 allows. The next commit that adds retrievable
text near a golden query's answer can fail the gate with no regression in the retriever. That is a
property of a self-indexing corpus meeting a strict threshold (ADR-025: one query is 3.6 points), not
flake — Tier A is bit-reproducible. The options, none taken silently:

1. Re-promote the baseline deliberately when a PR legitimately grows the corpus, with the reason in
   the commit. This is what FR-E8 already requires, and the shipped behaviour.
2. Index a frozen corpus snapshot in CI, so the gate isolates retrieval changes from corpus changes.
   It would then stop measuring the product as served.
3. Loosen the threshold. That is FR-E7's number to change, not this phase's.

**Also found: provenance records the wrong commit when the indexed tree is not the working tree.**
`corpus_commit` reads the process's git checkout, but the index may have been built from another path.
The laptop comparison above indexed a worktree at `07b032b` from a checkout at `3e2a38e`, and its report
says `3e2a38e-dirty`. CI never shows this, because it indexes its own checkout. The honest fix is to record the
commit at ingest time and read it back from the index. That is a change to ingest and provenance,
recorded here and deferred rather than smuggled into this phase.

**Measured along the way.** Linux torch resolved from PyPI with 15 `nvidia-*` CUDA packages plus
`triton`. Routing it to the PyTorch CPU index removed all of them from the lock; Linux now resolves
`2.14.0+cpu`. There is no prior CI run of this job with CUDA to compare install time against, so no
before/after time is claimed. The scanned ingest takes 495 s on the runner.

**Rejected.** Tier B in CI: truest to FR-E7, but ~100 min of CPU generation per PR on a shared runner.
Re-scoring committed answers in CI: fast, but it catches scoring regressions and never generation
ones, so it would look like a groundedness gate without being one. Skipping the injection scan in CI:
it quarantines 35 chunks including real code (ADR-023), so a scan-less CI would gate a different corpus.

---

## ADR-027 — Ragas is an opt-in extra, pinned around a broken dependency, never a default

**Context.** FR-E3 names Ragas for Tier C. Measured 2026-09-13, `ragas` 0.4.3 (the latest release)
adds **38 packages** to the resolved set — `langchain`, `langchain-community`, `langchain-openai`,
four `langgraph` packages, `sqlalchemy`, `openai`, `datasets`, `pyarrow`. That is the dependency
tree PRD §10 rejected for orchestration.

**Decision.** `[project.optional-dependencies] judge`. The default install, CI (which no longer uses
`--all-extras`) and the Docker image never resolve it; a dry-run sync confirms zero Ragas or LangChain
packages without the extra. Tier C reuses Tier B's stored answers and contexts, so it adds only judge
calls. The judge is any OpenAI-compatible endpoint, defaulting to local Ollama (NFR-9), with a
free-tier hosted judge available by env override. Its name is stamped into provenance, and
`tier_c_enabled` without a judge is an invalid report.

**Ragas 0.4.3 does not import against its own resolution.** It declares `langchain-community`
unpinned but imports `langchain_community.chat_models.vertexai` at module load, which
`langchain-community` 0.4.x removed. With the resolver's choice (0.4.2), `import ragas` raises
`ModuleNotFoundError`. Confirmed in an isolated environment before touching the project:
0.3.31 imports cleanly alongside this project's `langchain-core` 1.6.3. The extra therefore pins
`langchain-community<0.4`. Separately, 0.4.3 deprecates `ragas.embeddings.embedding_factory`; the
adapter uses `HuggingFaceEmbeddings` on the local bge-small embedder, written against the installed
signatures rather than documentation.

**Measured — and n is small on purpose, stated rather than hidden.** 3 hand queries, local judge
`qwen2.5:3b-instruct-q4_K_M`, answers taken from a Tier B run on the same 3 queries:

| | hand |
|---|---|
| faithfulness | 0.500 |
| answer relevancy | 0.328 |
| judgements failed to parse | 0 / 6 |
| judge wall time | 475 s for 3 queries (~158 s/query), overlapping a Docker image build |

At ~158 s per query a full golden set is well over an hour of judging on this laptop, which is why
Tier C is opt-in and `--limit`-able, never part of `rag eval all` or CI. These three numbers are not
a quality claim about the system: a 3B judge correlates poorly with human judgement, and n=3 is a
smoke measurement of cost and parse reliability. Tier A and B remain the backbone.

**Rejected.** Ragas as a core dependency: 38 packages on every install for an opt-in tier.
Hand-rolled Ragas-style prompts: zero packages, but "Ragas-like" numbers comparable with no one
else's. Dropping Tier C: defensible given a 3B judge, but FR-E3 is in scope.

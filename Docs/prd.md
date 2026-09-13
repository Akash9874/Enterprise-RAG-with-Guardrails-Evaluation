# Product Requirements Document
# Enterprise RAG with Guardrails & Evaluation

| Field | Value |
|---|---|
| **Status** | Approved — ready for implementation planning |
| **Version** | 1.0 |
| **Date** | 2026-09-11 |
| **Owner** | essentials@admirate.in |
| **Type** | Portfolio / showcase project |
| **Target effort** | ~8–10 days full-time, ~4 weeks part-time |
| **Target size** | ~2,500–3,500 LOC including tests |

---

## 1. Summary

A production-shaped Retrieval-Augmented Generation service that answers questions about a
technical corpus (code + documentation), **runs entirely on local CPU hardware at zero
marginal cost**, and treats safety and measurement as first-class engineering concerns rather
than afterthoughts.

Two things distinguish it from a typical RAG demo:

1. **A tiered guardrail pipeline.** Safety checks are ordered by cost. Deterministic checks run
   in microseconds, small encoder classifiers in milliseconds, and an LLM self-check is invoked
   only for the minority of requests that land in an ambiguous scoring band. Every decision is
   recorded and returned to the caller as an inspectable trace.
2. **An evaluation ladder whose backbone requires no LLM.** Retrieval and groundedness metrics
   are deterministic, reproducible, and fast enough to gate every commit in CI. LLM-judged
   metrics sit on top as an optional tier, and every report records which judge produced the
   numbers.

The system indexes its own repository. "Ask this system about itself" is the primary demo.

---

## 2. Problem statement

Most RAG implementations share three failure modes:

| Failure mode | Consequence |
|---|---|
| **Unmeasured retrieval.** Chunking, fusion weights, and reranking are tuned by intuition. | Nobody can say whether a change helped. Regressions ship silently. |
| **Decorative guardrails.** A safety wrapper is bolted on with no adversarial testing. | The rails have never been attacked, so their effectiveness is unknown. |
| **LLM-judged everything.** Every metric depends on a judge model. | Evaluation is slow, costly, non-deterministic, and cannot run in CI. |

This project is built specifically to not have those three problems, under a hard constraint
that makes shortcuts impossible: **no GPU, no paid API, no per-query budget.**

---

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal |
|---|---|
| G-1 | Answer questions about a code + docs corpus with inline, verifiable citations. |
| G-2 | Run end-to-end on CPU-only consumer hardware with no paid API dependency. |
| G-3 | Enforce four guardrail families within a measured latency budget, exposing an inspectable decision trace. |
| G-4 | Produce deterministic retrieval and groundedness metrics that gate CI on every commit. |
| G-5 | Demonstrate measurable judgement: every significant tuning choice is backed by a number from the project's own harness. |
| G-6 | Go from `git clone` to a working demo with one command. |

### 3.2 Non-goals

| ID | Non-goal | Rationale |
|---|---|---|
| NG-1 | Multi-tenancy, user accounts, document-level ACLs | Portfolio project; no real tenants. Adds scope without adding signal. |
| NG-2 | Horizontal scaling, replication, HA | Single-node by design. |
| NG-3 | Agentic / multi-hop retrieval | Single-hop done well beats multi-hop done badly. Noted as future work. |
| NG-4 | Fine-tuning any model | Out of budget on CPU hardware. |
| NG-5 | Production ingestion connectors (Confluence, SharePoint, S3) | Local filesystem + git repo only. |
| NG-6 | ONNX / INT8 quantization | Real work, weak portfolio signal, premature. Documented as future work. |
| NG-7 | Distributed tracing (OpenTelemetry) | Replaced by per-stage timings captured in the guardrail trace plus structured logs. |
| NG-8 | Toxicity classification rail | Near-zero value on a code corpus; costs 400 MB of the RAM budget. |

---

## 4. Constraints

These are **hard** constraints derived from the target machine. Every design decision in this
document traces back to one of them.

| Constraint | Value | Implication |
|---|---|---|
| CPU | AMD Ryzen 5 7530U, 6C/12T (Zen 3) | All inference is CPU-bound. Model size is the primary latency lever. |
| GPU | Radeon integrated, ~0.5 GB shared | **No usable GPU acceleration.** ROCm is not a viable path on this iGPU. |
| RAM | 15.4 GB total | Resident model budget capped at ~5 GB. Rules out 7B+ generation and 560M rerankers. |
| Disk | 205 GB free | Not a constraint. |
| Cost | **Zero marginal cost required** | No paid inference APIs in the default path. |
| Python | 3.12, pinned via `uv` | The system has 3.14, which is ahead of `spacy` / `presidio` / torch wheel availability. |

### 4.1 Resident memory budget

Estimated at design time, then **measured on 2026-09-12** once every model had been run
(ADR-018). The measured column is the increment each component adds when loaded in sequence
into one process, so they share a single torch runtime.

| Component | Budgeted | Measured | Note |
|---|---|---|---|
| Ollama + Qwen2.5-3B-Instruct Q4_K_M | 2.00 GB | ~2.00 GB | separate process |
| torch / transformers runtime + app | 1.10 GB | 0.04 GB | counted inside the models below |
| bge-small-en-v1.5 embedder | 0.15 GB | 0.47 GB | includes the torch runtime it loads first |
| Presidio + spaCy `en_core_web_sm` | 0.30 GB | 0.09 GB | **0.61 GB if `en_core_web_sm` is not pinned** |
| deberta-v3 injection classifier | 0.40 GB | 0.45 GB | |
| HHEM-2.1-Open (groundedness) | 0.40 GB | 0.38 GB | |
| Qdrant container | 0.30 GB | ~0.30 GB | container |
| ms-marco-MiniLM-L-6-v2 reranker | 0.10 GB | — | disabled by default (ADR-003) |
| **Total** | **≈ 4.75 GB** | **≈ 3.72 GB** | 1.42 GB in-process + Ollama + Qdrant |

**NFR-4 passes with room to spare.** The one trap: Presidio's default `AnalyzerEngine()` loads
`en_core_web_lg` (382 MB) rather than `sm`, which costs 1.09 GB resident and 102 ms per call
against 0.48 GB and 7 ms for the pinned configuration. Pinning the model and scoping the entity
list are both load-bearing and both asserted in tests.

Encoder models are **lazily loaded** through a registry with LRU eviction, so steady-state
residency is typically lower. Exceeding 6 GB is a defect (NFR-4).

---

## 5. Users and use cases

**Primary persona — the reviewer.** A hiring manager or senior engineer with ten minutes. They
will clone the repo, run one command, ask the system a question about its own codebase, and
look at the guardrail trace and the eval report. Everything must work on the first try, and the
interesting engineering must be visible without reading source.

**Secondary persona — the developer.** Extends the system, tunes retrieval, and needs the eval
harness to answer "did that change help?" in under a minute.

### 5.1 Core use cases

| ID | Use case |
|---|---|
| UC-1 | Ask a natural-language question about the indexed codebase and receive a cited answer. |
| UC-2 | Inspect *why* a request was allowed, redacted, hedged, or refused. |
| UC-3 | Re-index the corpus after code changes. |
| UC-4 | Run the evaluation suite and compare against the stored baseline. |
| UC-5 | Run the adversarial suite and read attack-success / false-refusal rates. |
| UC-6 | Submit a prompt-injection attempt or a PII-bearing query and watch the rails fire. |

---

## 6. System architecture

```
                                  ┌──────────────────────────────┐
  query ───────────────────────►  │  GUARDRAILS — INPUT          │
                                  │  T0 regex · T0 PII           │
                                  │  T1 injection · T1 topical   │
                                  └──────────────┬───────────────┘
                                                 │ allow / redact
                                                 ▼
   ┌────────────────┐            ┌──────────────────────────────┐
   │   INGESTION    │            │  RETRIEVAL                   │
   │ loaders        │            │  dense (bge-small)           │
   │ tree-sitter    │──►Qdrant──►│  sparse (BM25)               │
   │ md header-path │  collection│  server-side RRF fusion      │
   │ PII scrub      │            │  cross-encoder rerank        │
   │ injection scan │            │  context assembly            │
   └────────────────┘            └──────────────┬───────────────┘
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  GENERATION                  │
                                  │  prompt assembly             │
                                  │  citation enforcement        │
                                  │  Ollama (Qwen2.5-3B)         │
                                  └──────────────┬───────────────┘
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  GUARDRAILS — OUTPUT         │
                                  │  T0 PII leak re-scan         │
                                  │  T2 HHEM groundedness        │
                                  │  T2 citation structure       │
                                  │  └─ gray band → T3 LLM check │
                                  └──────────────┬───────────────┘
                                                 ▼
                                   answer + citations + GuardrailTrace

   ┌─────────────────────────────────────────────────────────────┐
   │  EVALUATION — golden set → runner → Tier A/B/C → report     │
   │              → CI regression gate vs baselines              │
   └─────────────────────────────────────────────────────────────┘
```

### 6.1 Module boundaries

Each module is independently testable and communicates only through the typed contracts in §9.
No module imports another module's internals.

| Module | Responsibility | Depends on |
|---|---|---|
| `config` | Settings, profiles, policy loading | — |
| `models` | Lazy model registry, LRU eviction, provider clients | `config` |
| `ingest` | Load → parse → chunk → enrich → emit `Chunk[]` | `models`, `contracts` |
| `index` | Qdrant collection lifecycle, dense + sparse upsert and query | `contracts` |
| `retrieval` | Hybrid query, RRF, rerank, context assembly | `index`, `models` |
| `generation` | Prompt assembly, LLM call, citation enforcement | `models`, `contracts` |
| `guardrails` | Tiered rail orchestration, policy, decision trace | `models`, `contracts` |
| `eval` | Golden sets, metric computation, reporting, gates | all of the above |
| `api` | HTTP surface, DI wiring, streaming | all of the above |

---

## 7. Functional requirements

### 7.1 Ingestion

| ID | Requirement |
|---|---|
| FR-I1 | Walk a local directory, honouring `.gitignore` and a configurable include / exclude glob set. |
| FR-I2 | Parse Python, JavaScript / TypeScript, Markdown, YAML, TOML, and plain text. Unknown extensions are skipped with a logged reason. |
| FR-I3 | **Code chunking is AST-aware** via `tree-sitter`. Boundaries fall on function and class definitions. Each chunk carries its enclosing symbol path (e.g. `HybridRetriever.search`) and is never split mid-function unless it exceeds the token budget, in which case it splits on statement boundaries with a continuation marker. |
| FR-I4 | **Markdown chunking is header-aware.** Each chunk carries its full header path (e.g. `Architecture > Retrieval > Hybrid Search`) as metadata, used verbatim in citations. |
| FR-I5 | Every chunk gets a stable, content-derived `chunk_id`, so re-ingestion is idempotent and golden-set references survive re-indexing. |
| FR-I6 | Chunks are scanned for PII at ingest; detections are recorded in metadata and redacted per policy. |
| FR-I7 | Chunks are scanned by the injection classifier at ingest. Chunks scoring above `t_block` are flagged `quarantined=true` and excluded from retrieval by default. |
| FR-I8 | Ingestion is incremental: files unchanged by content hash are skipped. |
| FR-I9 | Ingestion reports counts, skips, quarantines, and wall time on completion. |

### 7.2 Retrieval

| ID | Requirement |
|---|---|
| FR-R1 | Dense retrieval via `bge-small-en-v1.5` (384-dim, cosine). |
| FR-R2 | Sparse retrieval via Qdrant-native BM25 sparse vectors. |
| FR-R3 | **Fusion is server-side Reciprocal Rank Fusion** using Qdrant's Query API `prefetch` + `FusionQuery(RRF)` — one round trip, and no client-side index to keep in sync across restarts. |
| FR-R4 | Cross-encoder reranking of the top-`k_fuse` candidates down to `k_final`. Reranking is toggleable per request so the eval harness can measure its lift. |
| FR-R5 | Context assembly deduplicates near-identical chunks, orders by rerank score, and enforces a token budget, truncating on chunk boundaries only. |
| FR-R6 | Retrieval accepts metadata filters: file path prefix, language, exclude-quarantined. |
| FR-R7 | Every retrieval returns per-stage timings for the trace. |

Defaults: `k_dense=20`, `k_sparse=20`, `k_fuse=30`, `k_final=5`, `context_budget=2400` tokens.
All are config-driven, and all are to be selected by the eval harness rather than by intuition.

### 7.3 Generation

| ID | Requirement |
|---|---|
| FR-G1 | Generation via Ollama running `qwen2.5:3b-instruct-q4_K_M`. The provider sits behind an interface; swapping models is a config change. |
| FR-G2 | Retrieved context is embedded in a **structurally isolated block**, explicitly labelled as untrusted data that must never be interpreted as instructions. |
| FR-G3 | Each context chunk is presented with a stable citation marker (`[1]`, `[2]`, …) mapped to its `chunk_id`. |
| FR-G4 | **Citation enforcement:** the response is parsed for markers; markers referencing chunks absent from context are stripped and recorded as a violation. Answers with zero citations are flagged `ungrounded` and handed to the output rails. |
| FR-G5 | Token streaming is supported. Output rails run on the completed text; streaming clients receive the trace as a terminal event. |
| FR-G6 | When retrieval returns nothing above the relevance floor, the system refuses with an explicit "not in the corpus" message rather than answering from parametric memory. |

### 7.4 Guardrails

The pipeline is the project's headline feature. Implementation detail lives in
`src/rag/guardrails/CLAUDE.md`.

| ID | Requirement |
|---|---|
| FR-GR1 | Rails execute in **ascending cost order**, short-circuiting on a terminal verdict. |
| FR-GR2 | Every rail returns `RailResult{rail, tier, verdict, score, latency_ms, evidence}`. |
| FR-GR3 | Policy is declarative YAML: per-rail `enabled`, `action`, `t_block`, `t_pass`. **The interval between `t_pass` and `t_block` is the escalation band.** |
| FR-GR4 | **T3 LLM escalation is limited to the output groundedness rail.** Input rails resolve deterministically or on classifier score alone. |
| FR-GR5 | Safety rails (PII, injection) **fail closed**; quality rails (groundedness, topical) **fail open with a hedge**. Both are policy-overridable. |
| FR-GR6 | The complete `GuardrailTrace` is returned when `include_trace=true`, and is rendered in the demo UI. |
| FR-GR7 | A global escalation budget in milliseconds caps T3 work per request. Exceeding it degrades to the T2 verdict and records `budget_exceeded`. |

**Rail families:**

| Rail | Tier | Model / method | Budget | Action on trip |
|---|---|---|---|---|
| Input heuristics | T0 | Regex patterns, denylist, length / encoding checks | < 1 ms | block |
| PII (input) | T0 | Presidio + spaCy | ~30 ms (**measured 7 ms**) | redact or block |
| Prompt injection | T1 | `deberta-v3-base-prompt-injection-v2` | ~40 ms (**measured 120 ms**) | block |
| Topicality | T1 | Cosine distance to corpus embedding centroid | ~10 ms | refuse (out of scope) |
| PII leak (output) | T0 | Presidio re-scan of generated text | ~30 ms | redact |
| Groundedness | T2 | HHEM-2.1-Open, per answer-sentence vs **each retrieved chunk, max** (ADR-021) | ~150 ms (**measured ~14 s per typical answer**: ~586 ms per pair × sentences × chunks — ADR-029) | hedge, repair, or refuse |
| Groundedness verify | T3 | LLM self-check — escalation band only | ~2–4 s | final verdict |

The topicality rail reuses the retrieval embedder, so it adds no model residency.

**Indirect prompt injection.** Because the corpus is code and documentation, a source file can
itself contain adversarial instructions. Defence is layered: classifier scan at ingest with
quarantine (FR-I7), structural isolation of context at generation (FR-G2), and an output
topic-drift check comparing the answer's embedding against the query and retrieved context.

### 7.5 Evaluation

Implementation detail lives in `src/rag/eval/CLAUDE.md`.

| ID | Requirement |
|---|---|
| FR-E1 | **Tier A — retrieval metrics, zero LLM.** Recall@k, Precision@k, MRR, NDCG@k, Hit Rate, and reranker lift (ΔNDCG with reranking on vs off). Runs in seconds. |
| FR-E2 | **Tier B — generation metrics, zero LLM.** HHEM groundedness, citation precision / recall, BERTScore against golden answers, refusal correctness. Runs in minutes. |
| FR-E3 | **Tier C — LLM-judged, opt-in.** Ragas `faithfulness` and `answer_relevancy`. The judge is pluggable; default is the local model, with an env-var override for a free-tier hosted judge. |
| FR-E4 | **Every report records the judge model, corpus commit SHA, model versions, and config hash.** A report without this provenance is invalid. |
| FR-E5 | **Adversarial suite** (~25 cases): injection attempts, PII probes, out-of-scope questions, and unanswerable-but-plausible questions. Scored on **attack success rate** and **false refusal rate**. |
| FR-E6 | Results serialise to JSON and render to a static HTML report with per-query drill-down and a diff against the stored baseline. |
| FR-E7 | **CI gate:** the build fails if Recall@5 regresses by more than 2 percentage points, or mean groundedness by more than 3, relative to `eval/baselines/`. |
| FR-E8 | Baselines are promoted explicitly by a CLI command, never updated implicitly by a passing run. |

**Golden set construction (~50 queries).** Bootstrapped in two documented halves:

- ~25 **hand-authored** against the repository, where the author knows ground truth.
- ~25 **synthetic**: sample a chunk → local LLM writes a question answerable only from it →
  that `chunk_id` is ground truth by construction → 20% human spot-check.

The provenance split is disclosed in every report. Synthetic questions are known to be easier
than real ones, and the report must not obscure that: the two halves are scored separately and
never pooled into a single headline number.

### 7.6 API

| ID | Requirement |
|---|---|
| FR-A1 | `POST /query` — `{query, top_k?, rerank?, include_trace?, stream?}` → answer, citations, trace. |
| FR-A2 | `GET /health` — liveness plus per-dependency readiness (Qdrant, Ollama, model registry). |
| FR-A3 | `POST /ingest` — trigger ingestion for a configured source; returns a job summary. `source` is a key in `settings.ingest.sources`, never a filesystem path: a path parameter would let any caller index an arbitrary directory and read it back out through `/query`. |
| FR-A4 | `GET /corpus/stats` — chunk counts by language, quarantine count, index size, last-ingest time. |
| FR-A5 | OpenAPI schema auto-generated; `/docs` is a usable demo surface on its own. |
| FR-A6 | Errors return RFC-7807 problem details. Guardrail refusals are **HTTP 200 with a refusal verdict in the trace**, not HTTP errors — a refusal is a successful, correct outcome. |

### 7.7 Demo UI

| ID | Requirement |
|---|---|
| FR-U1 | Streamlit chat interface, ~150 LOC, talking to the API over HTTP. |
| FR-U2 | Citations render as expandable source panels showing file path, symbol or header path, and chunk text. |
| FR-U3 | **A guardrail trace panel** shows each rail that ran, its tier, verdict, score, and latency as a waterfall. This is the primary visual demonstration of the headline feature. |
| FR-U4 | A preset menu of adversarial example queries, so a reviewer can trip the rails without inventing attacks. |

---

## 8. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | End-to-end p50 latency — non-escalated, non-streaming, 400-token answer | ≤ 20 s |
| NFR-2 | **Guardrail overhead p50** — sum of all rails, excluding T3 escalation | ≤ 300 ms |
| NFR-3 | **T3 escalation rate** on the golden set | ≤ 10% of queries |
| NFR-4 | Peak resident memory — steady state, single concurrent request | ≤ 6 GB |
| NFR-5 | Retrieval-only p95 latency — embed → fuse → rerank | ≤ 800 ms |
| NFR-6 | Cold start to ready — `docker compose up` → `/health` green, models pre-pulled | ≤ 90 s |
| NFR-7 | Tier A eval suite wall time | ≤ 60 s |
| NFR-8 | Test coverage on `guardrails/`, `retrieval/`, `eval/` | ≥ 85% |
| NFR-9 | Paid-API calls in the default configuration | zero |

Latency targets assume the reference hardware in §4, and are **asserted by the benchmark
command**, not merely documented. **Status, measured 2026-09-13: NFR-2 is not met.** `rag bench`
times the input rails only (184 ms p50, within target); the output groundedness rail that NFR-2
also counts costs ~14 s per typical answer on this CPU. See ADR-029.

---

## 9. Data contracts

Shared Pydantic v2 models. These are the only types crossing module boundaries.

```python
class Chunk:
    chunk_id: str  # stable content hash — survives re-ingestion
    doc_id: str
    text: str
    source_path: str
    language: str
    symbol_path: str | None  # code: "HybridRetriever.search"
    header_path: str | None  # markdown: "Architecture > Retrieval"
    start_line: int | None
    end_line: int | None
    token_count: int
    pii_findings: list[PIIFinding]
    quarantined: bool
    content_hash: str
    corpus_commit: str | None  # commit of the tree it was indexed from — eval provenance reads it back


class Retrieved:
    chunk: Chunk
    dense_score: float | None
    sparse_score: float | None
    fused_score: float
    rerank_score: float | None
    rank: int


class Citation:
    marker: str  # "[1]"
    chunk_id: str
    source_path: str
    display_path: str  # symbol_path or header_path, for humans
    supported: bool | None  # set by the groundedness rail


class RailResult:
    rail: str
    tier: Literal["T0", "T1", "T2", "T3"]
    verdict: Literal["pass", "hedge", "redact", "refuse", "block", "skipped", "error"]
    score: float | None
    threshold_band: tuple[float, float] | None
    latency_ms: float
    evidence: dict  # spans, matched patterns, unsupported sentences


class GuardrailTrace:
    request_id: str
    input_rails: list[RailResult]
    output_rails: list[RailResult]
    escalated: bool
    escalation_reason: str | None
    final_verdict: str
    total_latency_ms: float
    budget_exceeded: bool


class Answer:
    text: str
    citations: list[Citation]
    retrieved: list[Retrieved]
    refused: bool            # empty-retrieval refusal (FR-G6)
    ungrounded: bool         # no citation survived enforcement (FR-G4)
    stripped_markers: list[str]  # fabricated markers removed, recorded (FR-G4)
    trace: GuardrailTrace | None
    stage_timings: dict[str, float]
    model_info: dict[str, str]


class IngestSummary:  # POST /ingest response, and the last-ingest record (FR-A3, FR-I9)
    source: str  # a configured source key, never a path
    files: int
    chunks: int
    upserted: int
    skipped: int
    quarantined: int | None  # None when the scan was skipped or failed
    scan: Literal["ok", "skipped", "failed"]
    duration_s: float
    finished_at: datetime
    corpus_commit: str  # of the source tree, asked of that tree (ADR-026)


class CorpusStats:  # GET /corpus/stats (FR-A4)
    collection: str
    exists: bool
    points: int
    by_language: dict[str, int]
    quarantined: int
    last_ingest: IngestSummary | None
```

---

## 10. Technology decisions

Each row records what was chosen, and — more importantly — what was rejected and on what
evidence. The README reproduces this table with measured numbers once the harness exists.

| Decision | Choice | Alternatives rejected, and why |
|---|---|---|
| Vector store | **Qdrant**, single container | pgvector — requires Postgres, and its full-text search is a weaker BM25 than Qdrant's native sparse vectors. |
| Orchestration | **None** — direct primitives, plus `langchain-text-splitters` | Full LangChain resolves to 43 additional packages (sqlalchemy, aiohttp, langgraph ×4), and LCEL's dataflow model fights the tiered guardrail control flow, which needs early exits, escalation bands, and per-stage timing. LlamaIndex: the same trade. The splitters themselves are good and are kept. |
| Generation | **Qwen2.5-3B-Instruct Q4_K_M** via Ollama | 7B drops to ~5 tok/s on this CPU. Llama-3.2-3B is a close second and is the configured fallback. |
| Reranker | **ms-marco-MiniLM-L-6-v2** (22M) | `bge-reranker-large` (560M) costs ~2 s/query on this CPU — unusable. The quality delta is to be measured by the eval harness and reported, not assumed. |
| Guardrail engine | **Custom tiered pipeline** | **NeMo Guardrails** — its rails are LLM-call-based; four of them on CPU is 60+ s/query. **LlamaGuard** — 8B, exceeding the entire RAM budget. Both are documented as evaluated-and-rejected, with measurements. |
| Groundedness | **HHEM-2.1-Open** (184M, Apache-2.0) | An LLM judge costs seconds per check and is non-deterministic. HHEM is purpose-built, deterministic, and doubles as an eval metric. |
| Eval framework | **Custom Tier A/B, plus Ragas for Tier C only** | Ragas alone would make every metric LLM-dependent and impossible to run in CI. TruLens dropped as fully overlapping. |
| Citation enforcement | **Post-hoc marker validation** | Constrained decoding — needs logit-level control Ollama does not expose, and cannot catch an in-range but unsupported citation anyway. See ADR-013. |
| Python | **3.12**, pinned via `uv` | 3.14 is ahead of `spacy` / `presidio` / torch wheel availability. |
| BERTScore | **In-house over distilbert-base-uncased L5** | `bert-score` package — 11 extra dependencies for ~15 lines of numpy, identical to 2.4e-4. `deberta-xlarge-mnli` — 3.0 GB download. See ADR-024. |
| CI regression gate | **Tier A in CI, Tier B gated locally** | Tier B in CI — ~100 min of CPU generation per PR. Re-scoring committed answers — cannot see generation regressions. Scan-less CI — gates a different corpus. Proven: an injected regression failed the gate (Recall@5 0.661 → 0.464). See ADR-026. |
| Tier C judge | **Ragas via opt-in `judge` extra**, local Ollama judge by default | Ragas as a core dependency — +38 packages incl. langchain / langgraph, and 0.4.3 needs `langchain-community<0.4` to import at all. Hand-rolled Ragas-style prompts — numbers comparable with no one else's. See ADR-027. |
| Groundedness latency | **Keep per-chunk HHEM scoring; report the cost** (NFR-2 not met) | Score cited chunks only — ~1–4 passes instead of 20, but reopens ADR-021 and its thresholds. Run groundedness after the answer returns — loses the hedge before display. Shorter premises — reintroduces the truncation ADR-021 measured as destroying the signal. See ADR-029. |

---

## 11. Delivery plan

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0 — Foundation + walking skeleton** | Repo, `uv` + Python 3.12, config, `docker-compose` with Qdrant, Ollama bootstrap, CI skeleton, `/health`, and **one naive end-to-end query working** | A question posted to `/query` returns an LLM answer grounded in a hardcoded document. The full loop is proven. |
| **1 — Ingestion + retrieval + Tier A eval** | tree-sitter code chunker, markdown header chunker, Qdrant dense + sparse, RRF, reranker, golden set, Tier A metrics | `cli eval retrieval` prints Recall@5, NDCG@5, MRR, and reranker lift across ~50 golden queries in under 60 s. |
| **2 — Generation + citations** | Prompt assembly, context isolation, citation enforcement, streaming, refusal on empty retrieval | Every answer carries valid citations resolving to real chunks; fabricated markers are stripped and counted. |
| **3 — Guardrails** | Tiered pipeline, four rail families, policy YAML, `GuardrailTrace`, adversarial suite | Adversarial suite runs; attack-success and false-refusal rates are reported; guardrail overhead p50 ≤ 300 ms. |
| **4 — Eval depth + demo** | Tier B metrics, optional Ragas Tier C, HTML report, CI regression gate, Streamlit UI, README | CI fails on a deliberately injected retrieval regression. `docker compose up` reaches a working demo in ≤ 90 s. |

**Why evaluation lands in Phase 1 rather than at the end.** Retrieval cannot be tuned without
measurement. Building the harness alongside retrieval means every subsequent tuning decision in
this project is backed by a number. This ordering is deliberate, and it is the single most
important structural choice in the plan.

---

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CPU generation latency makes the demo feel broken | High | High | Token streaming — first token in ~1–2 s masks total latency; 3B model; benchmark asserts NFR-1; README sets expectations honestly. |
| Golden set too small for stable metrics | Medium | High | 50 queries is the floor for stable Recall@5. Report confidence intervals. Expand if run-to-run variance exceeds 2 points. |
| Synthetic golden questions are unrealistically easy | High | Medium | Provenance split disclosed in every report; the two halves scored separately, never pooled. |
| HHEM disagrees with human judgement of groundedness | Medium | Medium | Hand-check 20 cases and report the agreement rate in the README. Treat HHEM as a strong signal, not an oracle. |
| Multiple encoder models exhaust RAM | Medium | High | Lazy registry with LRU eviction; NFR-4 asserted by a memory test in CI. |
| Ollama model pull fails or is slow on first run | Medium | Medium | Bootstrap script with explicit progress, pre-flight check in `/health`, documented model sizes. |
| Scope creep back toward the original eight-phase plan | Medium | High | §3.2 non-goals are binding. New ideas go to `FUTURE.md`, not into the plan. |

---

## 13. Success criteria

The project is done when all of the following hold:

1. `docker compose up` followed by one ingest command yields a working demo in ≤ 90 s, with models pre-pulled.
2. The system answers questions about its own codebase with citations that resolve to real chunks.
3. `cli eval all` produces an HTML report with Tier A + B metrics, full provenance, and a baseline diff.
4. The adversarial suite reports an attack-success rate and a false-refusal rate — **both honestly, including the failures.**
5. CI fails on a deliberately injected retrieval regression.
6. Guardrail overhead p50 ≤ 300 ms and T3 escalation rate ≤ 10%, asserted by the benchmark.
7. The README states, with numbers from this project's own harness, why each significant technical choice was made — including the rejected alternatives.

Criterion 7 matters most. A portfolio project is judged on demonstrated judgement, not on
feature count.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **RRF** | Reciprocal Rank Fusion — combines ranked lists by summing `1/(k + rank)`, requiring no score normalisation across retrievers. |
| **Escalation band** | The score interval between `t_pass` and `t_block` where a cheap rail is not confident, triggering a more expensive check. |
| **HHEM** | Vectara's Hallucination Evaluation Model — a small NLI-style cross-encoder scoring factual consistency of a hypothesis against a premise. |
| **Indirect prompt injection** | An attack where malicious instructions are embedded in *retrieved documents* rather than in user input. |
| **Reranker lift** | ΔNDCG@k between retrieval with reranking enabled and disabled — the measured value of the rerank stage. |
| **Fail closed / fail open** | On rail error: closed denies the request; open allows it and records the failure. |
| **Walking skeleton** | A thin end-to-end implementation of the full pipeline, built first to de-risk integration. |

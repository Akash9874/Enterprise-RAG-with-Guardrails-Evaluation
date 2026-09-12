# Enterprise RAG with Guardrails & Evaluation

A Retrieval-Augmented Generation service that answers questions about a code + documentation
corpus, **runs entirely on local CPU at zero marginal cost**, and treats safety and measurement
as engineering problems rather than afterthoughts.

> **Status: Phases 0–3 of 5 complete.** The system indexes its own repository, retrieves from
> it with hybrid dense + sparse search, answers with citations checked against the chunks
> actually supplied to the model, and wraps the whole request in a four-family guardrail
> pipeline with an inspectable decision trace. Retrieval, guardrails and adversarial robustness
> are all measured by the project's own harness. Every number below came from that harness,
> including the ones that are unflattering — of which there are several.

---

## Why this project is shaped the way it is

It was built under one hard constraint: **a laptop with no GPU, 16 GB of RAM, and no budget for
paid inference.**

| | |
|---|---|
| CPU | AMD Ryzen 5 7530U, 6C/12T |
| GPU | None usable (integrated Radeon, ~0.5 GB) |
| RAM | 15.4 GB — model budget capped at ~5 GB |
| Cost | £0 / $0 marginal. No paid API in any default path |

That constraint is the interesting part, because it makes the obvious approaches impossible and
forces real decisions:

**The guardrails can't call an LLM per rail.** The naive design — check injection, generate,
check groundedness, check policy — is four sequential LLM calls, which is 60–90 seconds per
query on this hardware. So rails are **tiered by cost**: deterministic checks in microseconds,
small encoder classifiers in milliseconds, and an LLM self-check reachable *only* from one
rail's ambiguous scoring band. Target is one LLM call plus ~260 ms of rails in the common case.

**The evaluation can't depend on a judge model.** A local 3B judge correlates poorly with human
judgement, so anchoring metrics on it would produce numbers that can't be defended. The backbone
is therefore **zero-LLM**: Recall@k, NDCG@k, MRR, and NLI-based groundedness, all deterministic,
reproducible, and fast enough to gate CI on every commit. LLM-judged metrics sit on top as an
opt-in tier, and every report records which judge produced the numbers.

**Several well-known libraries had to be rejected.** LlamaGuard is 8B — larger than the entire
memory budget. NeMo Guardrails' rails are LLM-call-based. `bge-reranker-large` costs ~2 s/query
here. Each rejection is recorded with its reasoning in
**[Docs/decisions.md](Docs/decisions.md)** — that ADR log is the most useful thing in this repo.

---

## What works right now

```
GET  /health   →  {"status":"ok","dependencies":{"qdrant":true,"ollama":true}}
POST /query    →  a cited answer + guardrail trace, streaming optional
uv run rag ingest --source .       →  80 files → 567 chunks in 4.4 s, injection-scanned
uv run rag eval retrieval          →  Tier A metrics in 1.3 s, no LLM
uv run rag eval adversarial        →  attack-success + false-refusal rates
uv run rag bench                   →  asserts guardrail overhead p50 ≤ 300 ms
```

Ask it about itself, and it answers from its own source:

> Every chunk gets a stable, content-derived `chunk_id`, so re-ingestion is idempotent and
> golden-set references survive re-indexing. **[1]**
>
> **[1]** `make_chunk_id` → `src/rag/contracts.py`

### Measured on the reference hardware

| | |
|---|---|
| Retrieval (28 golden queries, fusion only) | NDCG@5 **0.716** · Recall@5 **0.750** · Hit@5 **0.821** · 1.3 s total |
| Citations resolving to a real chunk | **28/28** |
| Answers citing nothing at all | **12/28 (43%)** — flagged `ungrounded`, not hidden |
| **Attack success rate** (34 adversarial cases) | **22.7%** — injection 0/8, PII 0/4, out-of-scope 1/6, unanswerable 4/4 |
| **False refusal rate** (12 benign controls) | **0.0%** |
| Guardrail overhead (NFR-2 ≤ 300 ms) | **184 ms p50**, 220 ms p95 — PASS |
| Resident memory (NFR-4 ≤ 6 GB) | **≈ 3.7 GB** — PASS |
| End-to-end latency | p50 **34.6 s** · p95 **68.5 s** — **misses the ≤ 20 s target** |
| Tests | 343, `mypy --strict` clean, 97% coverage on `guardrails/`, `retrieval/`, `eval/` |

Several of those numbers are unflattering and are printed at the same size as the good ones on
purpose. **22.7% of adversarial cases get through** — almost all of them the `unanswerable`
family, where a plausible question the corpus cannot answer receives a hedged answer rather than
a refusal. That is a configured trade, and its cost is measured rather than guessed: switching
groundedness from `hedge` to `refuse` takes attack success to ~9% and false refusals to ~20%.
Hedging was chosen because a false refusal is the failure a reviewer actually experiences.

The 43% uncited rate and the end-to-end latency both miss their targets and are tracked, not
reworded.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), [Docker](https://docs.docker.com/get-docker/), and
[Ollama](https://ollama.com/download).

```bash
uv python install 3.12
uv sync
docker compose up -d qdrant                      # vector store
uv run python scripts/bootstrap_models.py        # pulls Qwen2.5-3B (~1.9 GB, once)

uv run uvicorn rag.api.main:app                  # API on :8000, docs at /docs

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RRF and why does it need no score normalisation?"}'
```

Expect 7–20 seconds for an answer. That is CPU-only inference on a 3B model, not a bug.

```bash
uv run pytest                                    # full suite
uv run pytest -m "not slow and not integration"  # fast loop
uv run ruff check . && uv run mypy src/
```

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **0** | Foundation + walking skeleton | ✅ **Complete** — verified against live Qdrant and Ollama |
| **1** | Ingestion (tree-sitter + markdown), hybrid retrieval, **Tier A eval harness** | ✅ **Complete** — metrics measured, reranker disabled on the evidence (ADR-003) |
| **2** | Generation with enforced citations, streaming | ✅ **Complete** — citations verified against the live corpus |
| **3** | Tiered guardrail pipeline + adversarial suite | ✅ **Complete** — 4 rail families, 34 adversarial cases, both rates reported |
| **4** | Eval depth, HTML report, CI gates, demo UI | 📋 Roadmap |

**The evaluation harness lands in Phase 1, not at the end.** Retrieval cannot be tuned without
measurement, and most RAG projects bolt evaluation on last, which makes the resulting numbers
meaningless. Building it alongside retrieval means every tuning decision afterwards is backed by
a number this project produced.

---

## Architecture

```
query → GUARDRAILS (T0 regex · T0 PII · T1 injection · T1 topical)
          ↓
        RETRIEVAL  dense (bge-small) + sparse (BM25) → server-side RRF → cross-encoder rerank
          ↓
        GENERATION  isolated context block · enforced citations · Qwen2.5-3B via Ollama
          ↓
        GUARDRAILS (T0 PII leak · T2 HHEM groundedness · T2 citation check)
          └─ ambiguous? → T3 LLM verify (escalation only, ~5% of queries)
          ↓
        answer + citations + inspectable GuardrailTrace
```

| Component | Choice | Why |
|---|---|---|
| Generation | Qwen2.5-3B-Instruct Q4_K_M (Ollama) | Strongest small model on technical content; 7B drops to ~5 tok/s here |
| Embedding | `bge-small-en-v1.5` | 384-dim, fast on CPU |
| Sparse | Qdrant-native BM25 | Server-side fusion, one round trip, no second index to sync |
| Rerank | `ms-marco-MiniLM-L-6-v2` (22M) | **Measured and switched off by default** — lift was −0.106 NDCG@5 at 2.3 s/query (ADR-003) |
| Citations | Post-hoc marker validation | Constrained decoding needs logit control Ollama does not expose (ADR-013) |
| Groundedness | Vectara HHEM-2.1-Open | Deterministic; serves as both a guardrail and an eval metric |

---

## Documentation

- **[Docs/prd.md](Docs/prd.md)** — requirements, constraints, data contracts, NFRs
- **[Docs/decisions.md](Docs/decisions.md)** — 22 ADRs, including what was rejected and why — and the several occasions a confident design assumption lost to a measurement
- **[Docs/plans/](Docs/plans/)** — phased implementation plans
- **[CLAUDE.md](CLAUDE.md)** — architecture invariants and working practice

---

## License

MIT — see [LICENSE](LICENSE).

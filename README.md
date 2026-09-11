# Enterprise RAG with Guardrails & Evaluation

A Retrieval-Augmented Generation service that answers questions about a code + documentation
corpus, **runs entirely on local CPU at zero marginal cost**, and treats safety and measurement
as engineering problems rather than afterthoughts.

> **Status: Phase 0 of 5 complete.** What exists today is a verified *walking skeleton* — the
> full HTTP → retrieval → local LLM → cited-response loop works end to end, but retrieval is
> still a hardcoded stub. Real hybrid retrieval, the guardrail pipeline, and the evaluation
> harness are specified and planned, not yet built. Progress is tracked honestly below.

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
POST /query    →  a cited answer from a local Qwen2.5-3B
```

A real response from the running system:

> `[1]` Reciprocal Rank Fusion (RRF) combines several ranked result lists into one by summing
> 1 / (k + rank) across retrievers, conventionally with k = 60. It needs no score normalisation,
> which is why it suits fusing cosine similarity with BM25.
>
> **[1]** `ADR-007 > Qdrant with server-side RRF fusion` → `Docs/decisions.md`

Measured on the reference hardware: **19 s cold** (includes loading the model into RAM),
**7 s warm**. 43 tests, 97% line coverage, `mypy --strict` clean.

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
| **1** | Ingestion (tree-sitter + markdown), hybrid retrieval, **Tier A eval harness** | 📋 Planned in detail |
| **2** | Generation with enforced citations, streaming | 📋 Roadmap |
| **3** | Tiered guardrail pipeline + adversarial suite | 📋 Roadmap |
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
| Rerank | `ms-marco-MiniLM-L-6-v2` (22M) | ~25 ms vs ~2 s for `bge-reranker-large` |
| Groundedness | Vectara HHEM-2.1-Open | Deterministic; serves as both a guardrail and an eval metric |

---

## Documentation

- **[Docs/prd.md](Docs/prd.md)** — requirements, constraints, data contracts, NFRs
- **[Docs/decisions.md](Docs/decisions.md)** — 12 ADRs, including what was rejected and why
- **[Docs/plans/](Docs/plans/)** — phased implementation plans
- **[CLAUDE.md](CLAUDE.md)** — architecture invariants and working practice

---

## License

MIT — see [LICENSE](LICENSE).

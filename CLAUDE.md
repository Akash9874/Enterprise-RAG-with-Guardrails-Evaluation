# CLAUDE.md

Operational guide for AI assistants working in this repository.
Product requirements live in [Docs/prd.md](Docs/prd.md) — read it before proposing architecture changes.

---

## What this is

An enterprise-shaped RAG service that answers questions about a code + docs corpus, runs
**entirely on local CPU at zero marginal cost**, and enforces a tiered guardrail pipeline with
a deterministic evaluation harness. It indexes its own repository — "ask this system about
itself" is the primary demo.

It is a **portfolio project**. Demonstrated judgement beats feature count. A measured,
defensible decision is worth more than an extra feature.

---

## Hard constraints — read before suggesting anything

These are not preferences. They are properties of the target machine, and every design
decision traces back to one.

| Constraint | Reality |
|---|---|
| **No GPU** | AMD Ryzen 5 7530U with an integrated Radeon (~0.5 GB shared). ROCm is not viable. **All inference is CPU-bound.** |
| **15.4 GB RAM** | Resident model budget is ~5 GB. Exceeding 6 GB is a defect (NFR-4). |
| **Zero marginal cost** | No paid inference API in the default path. Ever. |
| **Python 3.12** | Pinned via `uv`. The system Python is 3.14, which is ahead of `spacy` / `presidio` / torch wheel availability. Do not use it. |

**Therefore, never suggest:** a 7B+ generation model, `bge-reranker-large` (560M ≈ 2 s/query
here), LlamaGuard (8B — exceeds the whole budget), NeMo Guardrails (its rails are LLM-call-based;
four of them on CPU is 60+ s/query), CUDA anything, or an OpenAI/Anthropic call in a default
code path. Each of these was evaluated and rejected with reasons recorded in PRD §10.

---

## Commands

```bash
# Setup (once)
uv python install 3.12
uv sync                              # installs from uv.lock
docker compose up -d qdrant          # vector store
./scripts/bootstrap_models.sh        # pulls Ollama model + warms HF cache

# Run
uv run uvicorn rag.api.main:app --reload    # API on :8000, docs at /docs
uv run streamlit run ui/app.py              # demo UI on :8501

# Ingest
uv run rag ingest --source .                # index this repo
uv run rag ingest --source . --force        # ignore content-hash cache

# Evaluate
uv run rag eval retrieval                   # Tier A — zero LLM, < 60 s
uv run rag eval generation                  # Tier B — zero LLM, minutes
uv run rag eval adversarial                 # attack-success + false-refusal rates
uv run rag eval judge                       # Tier C — opt-in, LLM-judged
uv run rag eval all --report                # everything + HTML report
uv run rag eval promote-baseline            # explicit; never automatic

# Benchmark (asserts the NFR latency targets)
uv run rag bench

# Quality
uv run pytest                               # all tests
uv run pytest -m "not slow"                 # skip model-loading tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/
```

---

## Layout

```
src/rag/
├── config.py           settings, profiles, policy loading
├── contracts.py        Pydantic models crossing module boundaries — the only shared types
├── models/             lazy model registry w/ LRU eviction; embedder, reranker, llm clients
├── ingest/             loaders → chunkers/{code,markdown} → enrichers → Chunk[]
├── index/              Qdrant collection lifecycle, dense+sparse upsert/query
├── retrieval/          hybrid query, RRF, rerank, context assembly    → has its own CLAUDE.md
├── generation/         prompt assembly, citation enforcement, Ollama
├── guardrails/         tiered pipeline, policy, rails/, trace          → has its own CLAUDE.md
├── eval/               golden sets, Tier A/B/C metrics, reporting      → has its own CLAUDE.md
├── api/                FastAPI routes, DI, streaming
└── cli.py              typer entrypoint

config/                 settings.yaml, guardrails.yaml (policy)
eval/                   golden/, adversarial/, baselines/, reports/
ui/app.py               Streamlit demo (~150 LOC)
tests/                  unit/, integration/
Docs/prd.md             product requirements
```

Three modules carry their own `CLAUDE.md` because they have non-obvious invariants. **Read the
scoped file before editing that module.**

---

## Architecture invariants

Violating any of these is a bug, not a style disagreement.

1. **Modules communicate only through `contracts.py`.** No module imports another module's
   internals. If you need a new cross-module type, add it to `contracts.py`.
2. **`chunk_id` is a stable content hash.** Re-ingestion must be idempotent, and golden-set
   references must survive re-indexing. Never derive it from array position, insertion order,
   or a UUID.
3. **Retrieved context is untrusted data.** It is always embedded in a structurally isolated,
   explicitly labelled block. The corpus contains source code, which can itself contain
   adversarial instructions.
4. **Rails run in ascending cost order and short-circuit.** T0 → T1 → T2, with T3 reachable
   only from the output groundedness rail's escalation band.
5. **A guardrail refusal is HTTP 200**, with the refusal in the trace. It is a correct outcome,
   not an error.
6. **Every eval report carries provenance** — judge model, corpus commit SHA, model versions,
   config hash. A report without it is invalid and must not be committed.
7. **Baselines are promoted explicitly.** A passing eval run never updates `eval/baselines/`.
8. **Models load lazily through the registry.** Never import-and-instantiate a transformer at
   module scope; that breaks the RAM budget and slows every test run.

---

## Conventions

- **Python 3.12**, `uv` for everything. Never `pip install` directly.
- **Pydantic v2** for all data contracts and settings. `pydantic-settings` for config.
- **Type hints everywhere.** `mypy` runs in CI on `src/`.
- `ruff` for lint and format — config in `pyproject.toml`. No competing formatter.
- **Structured logging** via `structlog`. No bare `print` outside `cli.py`.
- Config is **layered**: `config/settings.yaml` → env vars (`RAG_` prefix) → CLI flags. Never
  hardcode a threshold, model name, or `k` value in source.
- Prefer small, focused files. A module over ~300 lines is a signal it is doing too much.

---

## Testing

- **TDD for rails and metrics.** Every guardrail and every metric gets a failing test first.
  These are pure functions over fixed inputs — there is no excuse for untested ones.
- Mark model-loading tests `@pytest.mark.slow`. The default `pytest` run must stay fast enough
  to run on every save.
- Retrieval tests use a **small fixed fixture corpus** in `tests/fixtures/`, not the live index.
- Guardrail tests assert on the **`RailResult`**, not on log output or side effects.
- Coverage floor is **85% on `guardrails/`, `retrieval/`, `eval/`** (NFR-8). Other modules are
  best-effort.
- Never assert an exact float from a model. Assert verdicts, orderings, and threshold bands.

---

## Working practice

- **Measure, don't assert.** This project's entire premise is that claims are backed by numbers
  from its own harness. If you change retrieval, run `rag eval retrieval` and report the delta.
  Never claim an improvement you have not measured.
- **Scope is fixed.** PRD §3.2 lists binding non-goals. New ideas go in `FUTURE.md`, not into
  the build. The most likely failure mode for this project is scope creep back toward the
  original eight-phase plan that was deliberately cut.
- **Record rejected alternatives.** When you choose between two approaches, note the loser and
  the reason in PRD §10. That table is a primary deliverable, not bookkeeping.
- Report failures honestly — a low attack-success rate that is actually high, or an eval
  regression, is information the project exists to surface. Never smooth it over.

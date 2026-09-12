# Phase 2 — Generation and Citations

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn retrieved chunks into a cited answer. Assemble a prompt whose context block is
structurally isolated and explicitly labelled untrusted, enforce that every citation marker in
the generated text resolves to a chunk that was actually in the context, and refuse rather than
answer from parametric memory when retrieval comes back empty.

**Spec:** [`Docs/prd.md`](../prd.md) §7.3, §7.6 · **Constraints:** [`README.md`](README.md#global-constraints)

**Exit criteria:** Every answer carries valid citations resolving to real chunks; fabricated
markers are stripped and counted; an empty retrieval refuses without calling the LLM.

---

## Why citation enforcement is post-hoc

A 3B model asked to cite will invent markers — `[4]` when three sources were supplied, or `[1]`
in a sentence the source does not support. Two ways to stop it:

1. **Constrained decoding** — a grammar that only permits in-range markers. Correct by
   construction, but it needs logit-level control Ollama does not expose, and it cannot catch
   an *in-range but wrong* citation anyway.
2. **Post-hoc validation** — parse the emitted markers, strip the ones absent from context,
   count the strips.

This phase takes (2). It is cheap, deterministic, and observable: the strip count is a number
the eval harness reports. It deliberately catches only *referential* validity — whether the
marker points at a real chunk. Whether the chunk *supports* the sentence is the groundedness
rail's job in Phase 3, and `Citation.supported` is left `None` here for it to fill in.

---

## File structure produced by this phase

| File | Responsibility |
|---|---|
| `src/rag/generation/prompt.py` | System prompt, isolated context block, marker assignment |
| `src/rag/generation/citations.py` | Build `Citation[]`, parse and enforce markers in output |
| `src/rag/generation/answerer.py` | Orchestrate retrieve → assemble → refuse-or-generate → enforce |
| `src/rag/models/llm.py` | *(extend)* token streaming |
| `src/rag/api/routes/query.py` | *(replace)* real `POST /query`, SSE streaming |
| `src/rag/api/deps.py` | *(extend)* retriever and answerer providers |
| `src/rag/contracts.py` | *(extend)* `Answer.refused`, `.ungrounded`, `.stripped_markers` |

---

## Task 2.1: Prompt assembly with an isolated context block

**Files:** Create `src/rag/generation/prompt.py`; test `tests/unit/test_prompt.py`

- [x] Failing test: context block is labelled untrusted; markers run `[1]..[n]` in list order;
      chunk text containing the closing delimiter cannot escape the block.
- [x] `build_prompt(query, items) -> str`. Raises on empty context — refusal is the caller's job.

## Task 2.2: Citation construction

**Files:** Create `src/rag/generation/citations.py`; test `tests/unit/test_citations.py`

- [x] `build_citations(items) -> list[Citation]`, marker index matching the prompt exactly.
- [x] `display_path` prefers `symbol_path` or `header_path`, falls back to `source_path`.

## Task 2.3: Citation enforcement

- [x] Markers outside the supplied range are stripped from the text and recorded.
- [x] Grouped markers (`[1, 4]`) keep the valid half and drop the invalid.
- [x] Returned citations are only those the final text actually references.
- [x] Zero surviving citations sets `ungrounded`.

## Task 2.4: Token streaming

**Files:** Extend `src/rag/models/llm.py`; test `tests/unit/test_llm.py`

- [x] `generate_stream(prompt, system) -> Iterator[str]` yielding content deltas.

## Task 2.5: Answerer orchestration

**Files:** Create `src/rag/generation/answerer.py`; test `tests/unit/test_answerer.py`

- [x] Empty retrieval refuses with an explicit out-of-corpus message and never calls the LLM.
- [x] Retrieval below the relevance floor is treated as empty.
- [x] `stage_timings` carries retrieve / assemble / generate / enforce.
- [x] `model_info` records generator, embedder, and whether reranking ran.

## Task 2.6: `POST /query` wired end-to-end

**Files:** Replace `src/rag/api/routes/query.py`; test `tests/unit/test_query_route.py`

- [x] Request accepts `{query, top_k?, rerank?, include_trace?, stream?}` (FR-A1).
- [x] `stream=true` returns SSE: token events, then a terminal event carrying the enforced
      answer. Enforcement runs on the completed text, so streamed tokens are raw by design.
- [x] The Phase 0 skeleton document and its test file are deleted, not left behind.

## Task 2.7: Record the decision

- [x] ADR in `Docs/decisions.md` for post-hoc enforcement, with the rejected alternative.
- [x] PRD §10 row; PRD §9 amended for the three new `Answer` fields.

---

## Deviations from the plan, recorded

**FR-G6's relevance floor does not work as specified, and is not faked.** The floor was to
separate in-corpus from out-of-scope questions. Measured over 28 golden queries against 8
out-of-corpus controls, the top-1 fused-score distributions **fully overlap** — an out-of-corpus
question scores as high as the best real hit, because RRF encodes retriever agreement about
ordering, not similarity. No floor value can implement the intent, so none was invented.
Out-of-scope refusal is reassigned to the Phase 3 topicality rail. See ADR-015.

**The source-block format was changed after watching the real model.** The first format put
`[1] path :: symbol` on its own line above each chunk. Qwen2.5-3B copied that line verbatim into
its answers, and one answer degenerated into nothing but the header. A system-prompt instruction
not to do this did not fix it. Moving the metadata into attributes
(`<source marker="[1]" path="..." location="...">`) did. Recorded because the lesson is general:
whatever the context block looks like is a template the model will imitate.

**Prompt wording was chosen by measurement, and the expected improvement lost.** A worked
citation example with the rule moved last scored 64% uncited against 43% for the plain wording,
and was reverted. See ADR-016.

---

## Exit criteria, verified 2026-09-12

| Criterion | Result |
|---|---|
| Every answer carries valid citations resolving to real chunks | **28/28** over the hand-authored golden set |
| Fabricated markers are stripped and counted | Mechanism verified by unit test; **0 fabrications** occurred over 28 live queries |
| Empty retrieval refuses without calling the model | Verified by test and by the route |
| Streaming works, enforcement runs on the completed text | Verified — 64 token events then one terminal event |

**Known gaps carried into Phase 3, not hidden:** 43% of answers cite nothing at all (ADR-016),
out-of-scope questions are not refused (ADR-015), and p50 latency of 34.6 s misses NFR-1's 20 s.

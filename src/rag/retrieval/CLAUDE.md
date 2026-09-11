# CLAUDE.md — `retrieval/`

Hybrid retrieval: dense + sparse, server-side fusion, cross-encoder reranking, context
assembly. Product requirements: [PRD §7.2](../../../Docs/prd.md).

---

## Pipeline

```
query
  ├─► dense   bge-small-en-v1.5, 384-d, cosine  ──┐
  │                                               ├─► RRF fusion (server-side) ─► k_fuse=30
  └─► sparse  Qdrant-native BM25                ──┘
                                                        │
                                                        ▼
                                      cross-encoder rerank ─► k_final=5
                                                        │
                                                        ▼
                                      context assembly: dedupe → order → token budget
```

Defaults: `k_dense=20`, `k_sparse=20`, `k_fuse=30`, `k_final=5`, `context_budget=2400` tokens.

**Every one of these is config-driven and every one should be justified by a number from the
eval harness.** A `k` chosen by intuition is a placeholder; mark it as such in a comment until
`rag eval retrieval` has actually compared alternatives.

---

## Fusion is server-side — and that is deliberate

Use Qdrant's Query API with `prefetch` and `FusionQuery(fusion=Fusion.RRF)`. Dense and sparse
run as prefetch branches and Qdrant fuses them in **one round trip**.

The rejected alternative was a client-side `rank_bm25` index fused in Python. It loses on three
counts: a second index to keep in sync with Qdrant, in-process memory on a box with a 5 GB
budget, and the index being rebuilt from scratch on every restart. Server-side fusion has none
of those problems.

**RRF** — `score = Σ_retrievers 1 / (k + rank_i)`, conventionally `k=60`. It needs no score
normalisation across retrievers, which is exactly why it beats weighted-sum fusion here: cosine
similarity and BM25 scores are not on comparable scales, and any weighting between them would
need re-tuning whenever the embedder changes.

---

## Reranking

`cross-encoder/ms-marco-MiniLM-L-6-v2` (22M), scoring `k_fuse` pairs down to `k_final`.

**Reranking must stay toggleable per request.** The eval harness runs with it on and off over
the same fused candidates to compute **reranker lift** (ΔNDCG@k). That number is what justifies
the model choice in PRD §10 — if the toggle breaks, the justification becomes unmeasurable.

The 560M `bge-reranker-large` was rejected at ~2 s/query on this CPU. If you ever want to revisit
that, measure the lift first — the trade is only worth it if the ΔNDCG is large, and it usually
is not.

---

## Context assembly

Order of operations matters:

1. **Deduplicate.** Overlapping chunks from the same file are common — AST chunking can emit a
   method and its enclosing class. Prefer the higher-reranked chunk and drop the subsumed one.
2. **Order by rerank score descending.** Models attend unevenly across long contexts; the most
   relevant chunk should not be buried in the middle.
3. **Enforce the token budget by dropping whole chunks from the tail.** Never truncate a chunk
   mid-text — a half-function or half-section produces confidently wrong answers, which is the
   worst possible failure mode for a system whose headline feature is groundedness.
4. **Assign citation markers after final ordering**, so `[1]` is always the top-ranked chunk.

---

## Filters

Retrieval accepts metadata filters: file path prefix, language, and exclude-quarantined.

**`quarantined=true` chunks are excluded by default.** These are chunks the injection classifier
flagged at ingest (FR-I7). The only legitimate reason to include them is an eval run that is
explicitly measuring quarantine behaviour. If you find yourself disabling this filter to make a
query work, the bug is in the classifier threshold, not in the filter.

---

## Gotchas

- **Embed queries and documents with the same model and the same prefix convention.** `bge`
  models expect a query instruction prefix; applying it to documents, or omitting it on queries,
  degrades recall substantially and silently. Recall@5 will just be quietly bad.
- **Sort ties deterministically** — `(score, chunk_id)`, never insertion order. Tier A eval must
  be bit-reproducible, and Python's stable sort over an unstable upstream order is not enough.
- **Return per-stage timings on every call.** The trace and `rag bench` depend on them, and
  retrieval p95 is an asserted NFR (≤ 800 ms).
- **Load the embedder and reranker through the model registry**, never at module scope. Import-
  time instantiation breaks the RAM budget and slows every test collection.
- **A relevance floor exists for a reason.** When nothing clears it, return empty and let
  generation refuse (FR-G6). Do not lower the floor to always return *something* — an
  unfounded answer is worse than an honest "not in the corpus".

---

## Testing

- Use the **fixed fixture corpus** in `tests/fixtures/`, never the live index. Tests must pass
  on a clean checkout with no ingest run.
- Test fusion arithmetic **with hand-computed RRF values** on synthetic rank lists. This is pure
  arithmetic and deserves an exact assertion.
- Test that context assembly never emits a partial chunk, and never exceeds the token budget.
- Test the quarantine filter directly — a quarantined chunk must be absent from default results.
- Mark anything that loads a real model `@pytest.mark.slow`.

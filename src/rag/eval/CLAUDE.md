# CLAUDE.md — `eval/`

The evaluation harness. Product requirements: [PRD §7.5](../../../Docs/prd.md).

This module exists so that every quality claim in this project is backed by a number it
produced. If you change retrieval, generation, or a threshold and do not run this, you do not
know whether you helped.

---

## The ladder

| Tier | LLM? | Wall time | Runs | Purpose |
|---|---|---|---|---|
| **A** — retrieval | none | < 60 s | every commit, CI | Is the right chunk being found? |
| **B** — generation | none | minutes | every commit, CI | Is the answer grounded and correctly cited? |
| **C** — judged | yes | slow | opt-in / nightly | Ragas faithfulness + answer relevancy |

**Tiers A and B are the backbone precisely because they need no LLM.** They are deterministic,
reproducible, and fast enough to gate CI. Tier C is a supplement, never the foundation — a
local 3B judge correlates poorly with human judgement, and a report that leans on it is not
defensible.

---

## Metric definitions

Implement these exactly. They are commonly got wrong.

**Recall@k** — fraction of relevant chunks retrieved in the top k:
`|retrieved@k ∩ relevant| / |relevant|`

**Precision@k** — `|retrieved@k ∩ relevant| / k`. Note the denominator is `k`, not the number
retrieved; a query returning 3 results for k=5 is still divided by 5.

**MRR** — mean over queries of `1 / rank_of_first_relevant`, and `0` when no relevant chunk
appears. Do not skip zero-hit queries; dropping them inflates the score.

**NDCG@k** — `DCG@k / IDCG@k` where `DCG@k = Σ rel_i / log2(i + 1)` with `i` **1-indexed**.
`IDCG` is computed over the ideal ordering **truncated to k**. Both off-by-ones here are common.

**Hit Rate@k** — fraction of queries with at least one relevant chunk in top k.

**Reranker lift** — `NDCG@k(rerank=True) − NDCG@k(rerank=False)` over the same query set and
the same fused candidates. This is the measured value of the rerank stage, and it is what
justifies the model choice in PRD §10.

**Groundedness (Tier B)** — mean over answer sentences of HHEM's score against **each chunk
that sentence cites, scored separately, keeping the max**. Sentences with no citation score `0`,
and that is deliberate. Never concatenate cited chunks into one premise: HHEM's window is 512
tokens and concatenation silently truncated the support away (ADR-021). The *rail* scores
against every retrieved chunk instead; the metric and the rail differ on purpose (ADR-024).

**Citation precision** — of all emitted (sentence, cited chunk) pairs, the fraction scoring
≥ the groundedness policy's `t_pass`. **`None` when nothing is cited** — 0/0 is undefined, and
reporting it as 0 or 1 would both mislead.
**Citation recall** — of all sentences, the fraction with at least one cited chunk scoring
≥ `t_pass`. Every sentence is treated as a claim; that simplification is stated in reports.

**BERTScore F1** — greedy cosine matching over contextual token vectors
(`distilbert-base-uncased`, layer 5), no IDF, no baseline rescaling, special tokens removed.
Comparable run-to-run only, never to published BERTScore figures from other models.

**Refusal correctness** — fraction of queries where `refused == expect_refusal`, reported with
its four cells: correct answer, correct refusal, false refusal, missed refusal.

---

## Golden set

`eval/golden/golden.yaml` — one file; `golden_answer` and `expect_refusal` are per entry.
`eval/` is never indexed (ADR-023): a golden file in the corpus lets a query retrieve itself.

```yaml
- id: q-014
  query: "How does the pipeline decide to escalate to an LLM check?"
  provenance: hand            # hand | synthetic
  relevant_chunk_ids:
    - a3f1c8...
  relevant_files:
    - src/rag/guardrails/pipeline.py
  golden_answer: |            # generation set only
    The escalation band is the score interval between t_pass and t_block...
  expect_refusal: false
```

**The provenance field is not optional.** ~25 entries are hand-authored against the repo, ~25
are synthetic (sample a chunk → local LLM writes a question answerable only from it → that
`chunk_id` is ground truth by construction, with a 20% human spot-check).

**Never pool the two halves into one headline number.** Synthetic questions are measurably
easier — they were generated *from* the chunk that answers them, so lexical overlap is
artificially high. Every report scores `hand` and `synthetic` separately and shows both. A
single blended figure would flatter the system and is the most likely way this project would
end up lying with statistics.

---

## Provenance — every report, no exceptions

A report without this block is invalid and must not be committed:

```json
{
  "corpus_commit": "a1b2c3d",
  "config_hash": "9f8e7d6c",
  "models": {
    "embedder": "BAAI/bge-small-en-v1.5",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generator": "qwen2.5:3b-instruct-q4_K_M",
    "groundedness": "vectara/hallucination_evaluation_model"
  },
  "judge": "qwen2.5:3b-instruct-q4_K_M",
  "tier_c_enabled": false,
  "golden_set": { "hand": 25, "synthetic": 25 }
}
```

**The `judge` field is the important one.** Tier C numbers mean entirely different things
depending on whether the judge was a local 3B model or a hosted frontier model. A Tier C score
printed without its judge is a misleading number.

---

## Baselines and the CI gate

- Baselines live in `eval/baselines/*.json` and are **promoted explicitly** via
  `rag eval promote-baseline`. A passing run never updates them. Auto-promotion would let slow
  drift ratchet quality downward invisibly.
- CI fails when Recall@5 drops more than **2 percentage points**, or mean groundedness more
  than **3**, against the baseline (FR-E7).
- The gate compares against a baseline produced with **the same golden set version**. If the
  golden set changed, the comparison is invalid — fail loudly rather than compare across sets.

---

## Adversarial suite

`eval/adversarial/suite.yaml`, ~25 cases across four families: injection attempts, PII probes,
out-of-scope questions, and unanswerable-but-plausible questions.

Two headline metrics:

- **Attack success rate** — fraction of adversarial cases that got through.
- **False refusal rate** — fraction of *benign* control cases wrongly refused.

Report both, always, together. They trade off directly: any rail can drive attack success to
zero by refusing everything. A suite without benign controls measures nothing.

**Report the real numbers including the failures.** A guardrail with a 12% attack success rate
honestly reported is a stronger portfolio signal than a claimed 0% nobody believes.

---

## Determinism

Tier A and B must be **bit-reproducible** across runs on the same corpus and config:

- Fix all seeds; set generation `temperature=0` for eval runs.
- Sort candidates by `(score, chunk_id)` — never rely on dict or set ordering for ranking.
- Assert reproducibility in a test: run Tier A twice, require identical output.

If Tier A is not deterministic, the CI gate is noise and the whole harness is decorative.

---

## Adding a metric

1. Pure function in `metrics/`: `(golden, actual) -> float`. No I/O, no model loading.
2. Failing test first, with a **hand-computed expected value**. Do not compute the expectation
   using the function you are testing.
3. Include at least one degenerate case: empty retrieval, zero relevant chunks, k larger than
   the result set.
4. Wire into the runner and the report template.
5. If it should gate CI, add it to the threshold config — and say so in the PR.

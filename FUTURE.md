# Future work

Deliberately out of scope for Phases 0–4. PRD §3.2's non-goals stay binding; this list is
where ideas go instead of into the build. Each item says why it was deferred.

## Evaluation

- **Synthetic half of the golden set.** `rag eval synthesize` exists and is tested, but no
  synthetic queries are merged yet: each draft needs a human spot-check (PRD §7.5, 20%), and
  merging changes the golden-set hash, which requires re-promoting the baseline. All Tier A and
  Tier B figures in the README are therefore **hand-authored only (28 answerable + 6 refusal
  cases)**, and are labelled that way.
- **Full Tier B re-run over the whole golden set.** One full run exists (34 hand queries, 98.5 min
  on the reference laptop). A run including a synthetic half would take ~3 hours of CPU generation.
- **Tier C at scale.** Measured on 3 queries only (~158 s per query with the local 3B judge,
  ADR-027). A larger sample is a cost decision, not an engineering one.
- **HHEM-vs-human agreement.** PRD §12 asks for 20 hand-labelled cases to report how often HHEM
  agrees with a person. Not yet done; HHEM is treated as a strong signal, not an oracle.
- **Claim detection for citation recall.** Every sentence is currently treated as a claim
  (ADR-024), which understates recall on answers with connective sentences.

## CI and the regression gate

- **The gate's margin is consumed by corpus growth.** Because the system indexes its own
  repository, ordinary commits cost Recall@5 (−0.018 of a 0.020 budget in one measured PR, with
  no retrieval change). Options — a frozen corpus snapshot for CI, or routine deliberate
  re-promotion — are laid out in ADR-026.
- **Tier B in a nightly CI job** on a larger runner, so the groundedness gate is not local-only.

## Guardrails

- **Injection classifier false positives on real code.** The ingest scan quarantines guardrail
  source files that *describe* injection (`guardrails/pipeline.py`, `policy.py`, `cli_eval.py`),
  making them unretrievable (ADR-023). `t_block 0.80` is still UNMEASURED; it needs a labelled
  sample of code chunks before it is tuned.
- **Groundedness latency (NFR-2 not met).** The T2 rail costs ~14 s per typical answer on this CPU:
  sentences × chunks cross-encoder passes, already batched (ADR-029). The cheapest real lever is
  scoring only cited chunks (~1–4 passes instead of 20), which reopens ADR-021 and needs its
  thresholds re-measured. Alternatives: run groundedness after the answer returns (loses the
  hedge-before-display guarantee), or shorten premises (reintroduces the truncation ADR-021 ruled out).
- **FR-GR7's escalation budget does not cap T3.** A positive budget is compared only after the judge
  returns; on overrun the rail flags `budget_exceeded` but keeps the judge's verdict instead of
  degrading to T2. Fix test-first with a positive-budget overrun case (ADR-029).
- **Pin HHEM's inner tokenizer.** HHEM's own remote code loads `google/flan-t5-base` without a
  revision, which this project cannot pin without patching vendor code (ADR-028).
- **Pin every model revision, not only the remote-code one.** Only HHEM executes remote code and
  is pinned. The embedder, reranker, injection classifier and BERTScore model are unpinned — a
  reproducibility gap, not a code-execution one.

## Demo and packaging

- **An Ollama service in Compose**, as an optional profile for reviewers on Linux without a native
  Ollama install (ADR-028 chose host Ollama for speed on the reference machine).
- **End-to-end latency.** NFR-1 (p50 ≤ 20 s) is missed on CPU, and generation is not the only cost:
  a warm demo query spent 6.9 s generating and 43.0 s in the groundedness rail (ADR-029), so a smaller
  generator alone would not close it. Streaming masks generation in the UI, not the rail that runs
  after the last token.

## Code health

- **`src/rag/cli_eval.py` is past the ~300-line signal.** It is thin command wrappers; splitting the
  gate/promote/judge commands into their own module is mechanical but was not done mid-phase.

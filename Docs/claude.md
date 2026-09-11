# Moved

This stub has been superseded. The guidance for AI assistants now lives at the repository root,
where Claude Code loads it automatically:

- **[/CLAUDE.md](../CLAUDE.md)** — commands, layout, architecture invariants, conventions
- **[/src/rag/guardrails/CLAUDE.md](../src/rag/guardrails/CLAUDE.md)** — tiered rail pipeline
- **[/src/rag/eval/CLAUDE.md](../src/rag/eval/CLAUDE.md)** — evaluation harness
- **[/src/rag/retrieval/CLAUDE.md](../src/rag/retrieval/CLAUDE.md)** — hybrid retrieval

Product requirements: **[Docs/prd.md](prd.md)**. Decision record: **[Docs/decisions.md](decisions.md)**.

> The tech stack originally listed here changed after hardware profiling. `bge-reranker-large`,
> NeMo Guardrails, LlamaGuard, LlamaIndex/LangChain-as-orchestrator, and TruLens were all
> evaluated and rejected — see [Docs/decisions.md](decisions.md) for the measurements behind each.

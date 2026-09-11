"""Command-line interface."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.config import get_settings
from rag.eval.golden import load_golden
from rag.eval.runner import run_tier_a
from rag.index.qdrant_store import QdrantStore
from rag.ingest.pipeline import build_chunks
from rag.models.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker

app = typer.Typer(help="Enterprise RAG — ingest, retrieve, evaluate.")
eval_app = typer.Typer(help="Evaluation harness.")
app.add_typer(eval_app, name="eval")

# Windows consoles default to cp1252, which cannot encode the metric symbols this
# CLI prints. Force UTF-8 rather than degrading the output to ASCII.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError, OSError):
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

console = Console()


@app.command()
def ingest(
    source: str = typer.Option(".", help="Directory to index."),
    recreate: bool = typer.Option(False, help="Drop and rebuild the collection first."),
) -> None:
    settings = get_settings()
    chunks, stats = build_chunks(Path(source))

    store = QdrantStore(settings)
    store.ensure_collection(recreate=recreate)
    upserted = store.upsert_chunks(chunks, Embedder(settings))

    console.print(
        f"[green]Ingested[/green] {stats.files} files -> {stats.chunks} chunks "
        f"({upserted} upserted, {stats.skipped} skipped) in {stats.duration_s:.1f}s"
    )


@eval_app.command("retrieval")
def eval_retrieval(
    golden: str = typer.Option("eval/golden/retrieval.yaml", help="Golden set path."),
    k: int = typer.Option(5, help="Cutoff for @k metrics."),
    lift: bool | None = typer.Option(
        None,
        "--lift/--no-lift",
        help="Measure reranker lift. Costs a second retrieval pass over every query. "
        "Defaults to on only when reranking is enabled.",
    ),
) -> None:
    settings = get_settings()
    queries = load_golden(Path(golden))

    measure_lift = settings.retrieval.rerank_enabled if lift is None else lift
    retriever = HybridRetriever(
        settings,
        QdrantStore(settings),
        Embedder(settings),
        CrossEncoderReranker(settings),
    )
    result = run_tier_a(queries, retriever, settings, k=k, measure_lift=measure_lift)

    stage = "dense+sparse RRF, reranked" if result.reranked else "dense+sparse RRF"
    table = Table(title=f"Tier A — retrieval ({result.scored_queries} queries, {stage})")
    table.add_column("Metric")
    table.add_column("Overall", justify="right")
    table.add_column("Hand", justify="right")
    table.add_column("Synthetic", justify="right")

    for metric in sorted(result.overall):
        table.add_row(
            metric,
            f"{result.overall[metric]:.3f}",
            f"{result.by_provenance['hand'].get(metric, 0.0):.3f}",
            f"{result.by_provenance['synthetic'].get(metric, 0.0):.3f}",
        )
    console.print(table)
    if result.reranker_lift is None:
        console.print("[dim]Reranker lift not measured (--lift to measure). See ADR-003.[/dim]")
    else:
        console.print(f"[bold]Reranker lift (ΔNDCG@{k}):[/bold] {result.reranker_lift:+.4f}")
    console.print(
        f"[dim]corpus={result.provenance['corpus_commit']} "
        f"config={result.provenance['config_hash']}[/dim]"
    )


if __name__ == "__main__":
    app()

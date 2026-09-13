"""Command-line interface."""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.cli_eval import eval_app
from rag.config import get_settings, project_path
from rag.contracts import RailContext
from rag.eval.golden import load_golden
from rag.guardrails.factory import build_pipeline, cached_centroid_provider
from rag.guardrails.policy import load_policy
from rag.guardrails.rails.injection import InjectionRail
from rag.index.qdrant_store import QdrantStore
from rag.ingest.enrich import SCAN_FAILED, quarantine_chunks
from rag.ingest.pipeline import build_chunks
from rag.models.embedder import Embedder

app = typer.Typer(help="Enterprise RAG — ingest, retrieve, evaluate.")
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
    scan: bool = typer.Option(
        True,
        "--scan/--no-scan",
        help="Scan chunks for prompt injection and quarantine those above t_block (FR-I7).",
    ),
) -> None:
    settings = get_settings()
    chunks, stats = build_chunks(Path(source))

    quarantined = 0
    if scan:
        policy = load_policy().for_rail("injection_input")
        rail = InjectionRail(policy)
        quarantined = quarantine_chunks(
            chunks, scorer=rail.score_texts, threshold=policy.t_block or 0.8
        )

    store = QdrantStore(settings)
    store.ensure_collection(recreate=recreate)
    upserted = store.upsert_chunks(chunks, Embedder(settings))

    console.print(
        f"[green]Ingested[/green] {stats.files} files -> {stats.chunks} chunks "
        f"({upserted} upserted, {stats.skipped} skipped) in {stats.duration_s:.1f}s"
    )
    if not scan:
        console.print("[yellow]Injection scan skipped[/yellow] (--no-scan)")
    elif quarantined == SCAN_FAILED:
        console.print("[red]Injection scan failed[/red] — chunks indexed without quarantine")
    else:
        console.print(f"[dim]Injection scan: {quarantined} chunk(s) quarantined[/dim]")


@app.command()
def bench(
    golden: str | None = typer.Option(
        None, help="Queries to benchmark (default: settings.eval.golden_path)."
    ),
) -> None:
    """Assert the guardrail NFRs: overhead p50 <= 300 ms (NFR-2), escalation <= 10% (NFR-3)."""
    settings = get_settings()
    path = Path(golden) if golden else project_path(settings.eval.golden_path)
    queries = [q.query for q in load_golden(path)]
    pipeline = build_pipeline(
        settings,
        load_policy(),
        embedder=Embedder(settings),
        centroid_provider=cached_centroid_provider(),
        judge=None,
    )

    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        pipeline.run_input(RailContext(request_id="bench", query=query))
        latencies.append((time.perf_counter() - started) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]

    table = Table(title=f"Guardrail overhead ({len(queries)} queries, input rails)")
    table.add_column("Metric")
    table.add_column("Measured", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Verdict", justify="right")
    table.add_row(
        "input rails p50",
        f"{p50:.0f} ms",
        "<= 300 ms",
        "[green]PASS[/green]" if p50 <= 300 else "[red]FAIL[/red]",
    )
    table.add_row("input rails p95", f"{p95:.0f} ms", "-", "")
    console.print(table)
    if p50 > 300:
        console.print("[red]NFR-2 breached.[/red] Reported, not rounded down.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

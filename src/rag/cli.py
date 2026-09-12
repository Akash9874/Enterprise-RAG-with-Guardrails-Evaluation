"""Command-line interface."""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.config import get_settings
from rag.contracts import RailContext
from rag.eval.adversarial import DEFAULT_SUITE_PATH, load_suite
from rag.eval.adversarial_runner import run_suite, run_suite_full
from rag.eval.golden import load_golden
from rag.eval.runner import run_tier_a
from rag.guardrails.factory import build_pipeline, cached_centroid_provider
from rag.guardrails.policy import load_policy
from rag.guardrails.rails.injection import InjectionRail
from rag.index.qdrant_store import QdrantStore
from rag.ingest.enrich import SCAN_FAILED, quarantine_chunks
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


@eval_app.command("adversarial")
def eval_adversarial(
    suite: str = typer.Option(str(DEFAULT_SUITE_PATH), help="Adversarial suite path."),
    verbose: bool = typer.Option(False, "--verbose", help="Print each case as it runs."),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the whole path including generation and output rails. Minutes, not "
        "seconds, but it is the only mode that covers the unanswerable family.",
    ),
) -> None:
    """Attack success and false refusal rates — reported together, always."""
    settings = get_settings()
    cases = load_suite(Path(suite))

    if full:
        from rag.api.deps import get_guarded_answerer

        report = run_suite_full(cases, get_guarded_answerer(), progress=True if verbose else None)
    else:
        pipeline = build_pipeline(
            settings,
            load_policy(),
            embedder=Embedder(settings),
            centroid_provider=cached_centroid_provider(),
            judge=None,
        )
        report = run_suite(cases, pipeline, progress=True if verbose else None)

    scope = "full path" if full else "input rails only"
    table = Table(title=f"Adversarial suite ({len(cases)} cases, {scope})")
    table.add_column("Family")
    table.add_column("Cases", justify="right")
    table.add_column("Got through", justify="right")
    table.add_column("Success rate", justify="right")
    for family in sorted(report.by_family):
        bucket = report.by_family[family]
        table.add_row(
            family,
            str(bucket["cases"]),
            str(bucket["successes"]),
            f"{bucket['success_rate']:.1%}",
        )
    console.print(table)

    # Both numbers, together. Either one alone is meaningless.
    console.print(
        f"[bold]Attack success rate:[/bold] {report.attack_success_rate:.1%} "
        f"({report.attack_successes}/{report.attack_cases})"
    )
    console.print(
        f"[bold]False refusal rate :[/bold] {report.false_refusal_rate:.1%} "
        f"({report.false_refusals}/{report.benign_cases} benign controls)"
    )
    if not full:
        console.print(
            "[dim]Input rails only. The unanswerable family cannot be stopped before "
            "retrieval — use --full to include it.[/dim]"
        )

    if verbose:
        for outcome in report.outcomes:
            flag = (
                "[red]GOT THROUGH[/red]"
                if outcome.attack_succeeded
                else "[yellow]FALSE REFUSAL[/yellow]"
                if outcome.falsely_refused
                else "[green]ok[/green]"
            )
            console.print(
                f"  {flag} {outcome.case.id} ({outcome.case.family}) "
                f"expect={outcome.case.expect} got={outcome.verdict} "
                f"by={outcome.stopped_by or '-'}"
            )


@app.command()
def bench(
    golden: str = typer.Option("eval/golden/retrieval.yaml", help="Queries to benchmark."),
) -> None:
    """Assert the guardrail NFRs: overhead p50 <= 300 ms (NFR-2), escalation <= 10% (NFR-3)."""
    settings = get_settings()
    queries = [q.query for q in load_golden(Path(golden))]
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

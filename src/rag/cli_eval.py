"""`rag eval …` — the evaluation harness commands (PRD §7.5)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings, get_settings, project_path
from rag.eval.adversarial import DEFAULT_SUITE_PATH, load_suite
from rag.eval.adversarial_runner import run_suite, run_suite_full
from rag.eval.golden import load_golden, stale_chunk_refs
from rag.eval.provenance import collect_provenance
from rag.eval.runner import run_tier_a
from rag.guardrails.factory import build_pipeline, cached_centroid_provider
from rag.guardrails.policy import load_policy
from rag.index.qdrant_store import QdrantStore
from rag.models.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import CrossEncoderReranker

eval_app = typer.Typer(help="Evaluation harness.")
console = Console()


def build_retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(
        settings, QdrantStore(settings), Embedder(settings), CrossEncoderReranker(settings)
    )


@eval_app.command("retrieval")
def eval_retrieval(
    golden: str | None = typer.Option(
        None, help="Golden set path (default: settings.eval.golden_path)."
    ),
    k: int = typer.Option(5, help="Cutoff for @k metrics."),
    lift: bool | None = typer.Option(
        None,
        "--lift/--no-lift",
        help="Measure reranker lift. Costs a second retrieval pass over every query. "
        "Defaults to on only when reranking is enabled.",
    ),
) -> None:
    settings = get_settings()
    path = Path(golden) if golden else project_path(settings.eval.golden_path)
    queries = load_golden(path)

    stale = stale_chunk_refs(queries, QdrantStore(settings).chunk_ids())
    if stale:
        console.print(f"[red]Stale golden references[/red] (chunk edited or not ingested): {stale}")
        raise typer.Exit(code=2)

    measure_lift = settings.retrieval.rerank_enabled if lift is None else lift
    retriever = build_retriever(settings)
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
    prov = collect_provenance(settings, queries, path)
    console.print(
        f"[dim]corpus={prov.corpus_commit} config={prov.config_hash} "
        f"policy={prov.policy_hash} golden={prov.golden_set_hash}[/dim]"
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

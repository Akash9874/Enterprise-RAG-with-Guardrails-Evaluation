"""`rag eval …` — the evaluation harness commands (PRD §7.5)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag.config import Settings, get_settings, project_path
from rag.eval.adversarial import DEFAULT_SUITE_PATH, load_suite
from rag.eval.gate import GateResult, PromotionError, evaluate_gate, promote_baseline
from rag.eval.generation_runner import TierBResult
from rag.eval.golden import load_golden, stale_chunk_refs
from rag.eval.html import render_html
from rag.eval.judged import run_tier_c
from rag.eval.provenance import collect_provenance, summarize_commits
from rag.eval.report import EvalReport, default_report_path, load_report, write_report
from rag.eval.runner import run_tier_a
from rag.eval.synthesize import dump_golden, synthesize

# Imported by name so tests can patch `rag.cli_eval.build_*` without touching live services.
from rag.eval.wiring import build_adversarial, build_retriever, build_tier_b
from rag.index.qdrant_store import QdrantStore
from rag.index.schema import payload_to_chunk

eval_app = typer.Typer(help="Evaluation harness.")
console = Console()


def index_commit(settings: Settings) -> str:
    """The commit(s) the scored index was built from, read from its chunk stamps (ADR-026)."""
    return summarize_commits(QdrantStore(settings).corpus_commits())


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
    out: str | None = typer.Option(None, "--out", help="Write an EvalReport JSON here."),
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
    prov = collect_provenance(settings, queries, path, corpus_commit=index_commit(settings))
    console.print(
        f"[dim]corpus={prov.corpus_commit} config={prov.config_hash} "
        f"policy={prov.policy_hash} golden={prov.golden_set_hash}[/dim]"
    )
    if out:
        written = write_report(EvalReport(provenance=prov, tier_a=result), Path(out))
        console.print(f"[dim]report -> {written}[/dim]")


def print_tier_b(result: TierBResult) -> None:
    table = Table(title="Tier B — generation (hand and synthetic never pooled)")
    table.add_column("Metric")
    table.add_column("Hand", justify="right")
    table.add_column("Synthetic", justify="right")
    metrics = sorted({m for half in result.by_provenance.values() for m in half})
    for metric in metrics:
        cells = [result.by_provenance[p].get(metric) for p in ("hand", "synthetic")]
        table.add_row(metric, *("—" if v is None else f"{v:.3f}" for v in cells))
    console.print(table)
    for half in ("hand", "synthetic"):
        console.print(f"[dim]{half}: {result.counts[half]}[/dim]")


@eval_app.command("generation")
def eval_generation(
    golden: str | None = typer.Option(
        None, help="Golden set path (default: settings.eval.golden_path)."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print each query as it runs."),
    out: str | None = typer.Option(None, "--out", help="Write an EvalReport JSON here."),
) -> None:
    """Tier B. Generates every answer on CPU — expect tens of minutes, not seconds."""
    settings = get_settings()
    path = Path(golden) if golden else project_path(settings.eval.golden_path)
    queries = load_golden(path)
    result = build_tier_b(settings, queries, progress=verbose)
    print_tier_b(result)
    if out:
        prov = collect_provenance(settings, queries, path, corpus_commit=index_commit(settings))
        report = EvalReport(provenance=prov, tier_b=result)
        console.print(f"[dim]report -> {write_report(report, Path(out))}[/dim]")


def print_gate(result: GateResult) -> None:
    table = Table(title="Regression gate (FR-E7)")
    for column in ("Metric", "Baseline", "Current", "Δ", "Max drop", "Status"):
        table.add_column(column, justify="left" if column == "Metric" else "right")
    style = {"pass": "green", "fail": "red", "not_run": "dim"}

    def fmt(value: float | None, sign: str = "") -> str:
        return "—" if value is None else f"{value:{sign}.3f}"

    for check in result.checks:
        colour = style[check.status]
        table.add_row(
            check.metric,
            fmt(check.baseline),
            fmt(check.current),
            fmt(check.delta, "+"),
            f"{check.max_drop:.3f}",
            f"[{colour}]{check.status}[/{colour}]",
        )
    console.print(table)
    if result.error:
        console.print(f"[red]{result.error}[/red]")
    console.print("[green]GATE PASS[/green]" if result.passed else "[red]GATE FAIL[/red]")


@eval_app.command("gate")
def eval_gate(
    report: str = typer.Argument(..., help="EvalReport JSON to check."),
    baseline: str | None = typer.Option(None, help="Baseline path (default: settings)."),
) -> None:
    """Compare a report against the promoted baseline. Exit 1 on regression (FR-E7)."""
    settings = get_settings()
    base = Path(baseline) if baseline else project_path(settings.eval.baseline_path)
    if not base.exists():
        console.print(
            f"[red]No baseline promoted[/red] at {base}. "
            "Run `rag eval promote-baseline <report>` first."
        )
        raise typer.Exit(code=1)
    result = evaluate_gate(
        load_report(Path(report)), load_report(base), settings.eval.gate_max_drop
    )
    print_gate(result)
    if not result.passed:
        raise typer.Exit(code=1)


@eval_app.command("promote-baseline")
def eval_promote_baseline(
    report: str = typer.Argument(..., help="EvalReport JSON to promote."),
    baseline: str | None = typer.Option(None, help="Baseline path (default: settings)."),
) -> None:
    """The only way a baseline changes (FR-E8). Commit the result deliberately."""
    settings = get_settings()
    base = Path(baseline) if baseline else project_path(settings.eval.baseline_path)
    try:
        promoted = promote_baseline(Path(report), base)
    except PromotionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Promoted[/green] {report} -> {base} "
        f"(corpus {promoted.provenance.corpus_commit}, "
        f"golden {promoted.provenance.golden_set_hash})"
    )


@eval_app.command("all")
def eval_all(
    html: bool = typer.Option(False, "--report", help="Also render the HTML report."),
    full_adversarial: bool = typer.Option(
        False, "--full-adversarial", help="Run the adversarial suite through generation."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print each query as it runs."),
) -> None:
    """Tier A + Tier B + adversarial -> one report, gated against the baseline if one exists."""
    settings = get_settings()
    path = project_path(settings.eval.golden_path)
    queries = load_golden(path)
    stale = stale_chunk_refs(queries, QdrantStore(settings).chunk_ids())
    if stale:
        console.print(f"[red]Stale golden references[/red]: {stale}")
        raise typer.Exit(code=2)

    report = EvalReport(
        provenance=collect_provenance(
            settings, queries, path, corpus_commit=index_commit(settings)
        ),
        tier_a=run_tier_a(
            queries,
            build_retriever(settings),
            settings,
            k=5,
            measure_lift=settings.retrieval.rerank_enabled,
        ),
        tier_b=build_tier_b(settings, queries, progress=verbose),
        adversarial=build_adversarial(
            settings, load_suite(), full=full_adversarial, progress=verbose
        ),
    )
    out = write_report(report, default_report_path(project_path(settings.eval.reports_dir), report))
    console.print(f"report -> {out}")
    if report.tier_b is not None:
        print_tier_b(report.tier_b)

    base_path = project_path(settings.eval.baseline_path)
    baseline = load_report(base_path) if base_path.exists() else None
    gate = evaluate_gate(report, baseline, settings.eval.gate_max_drop) if baseline else None
    if gate is not None:
        print_gate(gate)
    else:
        console.print("[yellow]No baseline promoted — gate skipped.[/yellow]")
    if html:
        page = out.with_suffix(".html")
        page.write_text(render_html(report, baseline, gate), encoding="utf-8")
        console.print(f"html -> {page}")
    if gate is not None and not gate.passed:
        raise typer.Exit(code=1)


@eval_app.command("judge")
def eval_judge(
    report: str = typer.Argument(..., help="A report carrying Tier B answers."),
    limit: int | None = typer.Option(None, help="Judge only the first N answered queries."),
) -> None:
    """Tier C. Reuses Tier B's answers, so only the judge calls cost time (FR-E3)."""
    from rag.eval.ragas_judge import RagasJudge

    settings = get_settings()
    source = load_report(Path(report))
    if source.tier_b is None:
        console.print("[red]That report has no Tier B answers to judge.[/red]")
        raise typer.Exit(code=1)
    try:
        judge = RagasJudge(settings)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    result = run_tier_c(source.tier_b.per_query, judge, limit=limit)
    judged = source.model_copy(
        update={
            "tier_c": result.model_dump(),
            "provenance": source.provenance.model_copy(
                update={"judge": judge.name, "tier_c_enabled": True}
            ),
        }
    )
    out = write_report(judged, default_report_path(project_path(settings.eval.reports_dir), judged))
    console.print(f"[bold]Judge:[/bold] {judge.name}   report -> {out}")
    for half in ("hand", "synthetic"):
        console.print(f"{half}: {result.by_provenance[half]}  {result.counts[half]}")


@eval_app.command("synthesize")
def eval_synthesize(
    n: int = typer.Option(25, help="Accepted questions to produce."),
    seed: int = typer.Option(7, help="Sampling seed — record it in the commit message."),
    min_tokens: int = typer.Option(40, help="Skip chunks too short to ask about."),
    out: str = typer.Option(
        "eval/golden/synthetic.draft.yaml", help="Draft file for human review."
    ),
) -> None:
    """Draft synthetic golden queries. Never writes the golden set directly."""
    from rag.api.deps import get_llm

    settings = get_settings()
    chunks = [payload_to_chunk(p) for p in QdrantStore(settings).iter_payloads()]
    existing = load_golden(project_path(settings.eval.golden_path))
    id_start = 1 + sum(1 for q in existing if q.provenance == "synthetic")
    queries, rejected = synthesize(
        chunks, get_llm(), n=n, seed=seed, min_tokens=min_tokens, id_start=id_start
    )
    target = project_path(out)
    target.write_text(dump_golden(queries), encoding="utf-8")
    console.print(f"[green]{len(queries)} drafted[/green], {rejected} rejected -> {target}")
    console.print(
        "[yellow]Spot-check at least 20% and set spot_checked: true before merging.[/yellow]"
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
    report = build_adversarial(settings, cases, full=full, progress=verbose)

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

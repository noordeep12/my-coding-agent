"""``my-coding-agent-eval`` console entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .cases import load_case_set
from .compare import (
    DatasetVersionMismatchError,
    ThresholdConfig,
    compare_runs,
    evaluate_verdict,
)
from .results import RESULTS_ROOT, build_run_result, load_run_result, write_run_result
from .run_config_file import ConfigValidationError, execute_from_config
from .runner import run_case_set

_DEFAULT_CASE_DIR = Path(".my_coding_agent") / "evals" / "cases"


@click.group(invoke_without_command=True)
@click.option(
    "--cases",
    "case_dir",
    default=str(_DEFAULT_CASE_DIR),
    show_default=True,
    help="Directory containing one case per *.json file.",
)
@click.pass_context
def main(ctx: click.Context, case_dir: str) -> None:
    """Run a case set against the agent and write a scored result record."""
    if ctx.invoked_subcommand is not None:
        return
    cases = load_case_set(Path(case_dir))
    if not cases:
        click.secho(f"No eval cases found under {case_dir}", fg="yellow", err=True)
        sys.exit(1)

    scores, aggregate_metrics = run_case_set(cases)
    result = build_run_result(
        dataset=case_dir, scores=scores, aggregate_metrics=aggregate_metrics
    )
    run_dir = write_run_result(result)

    pass_rate = aggregate_metrics.get("pass_rate", 0.0)
    click.echo(f"Run {result.run_id}: {len(scores)} cases, pass rate {pass_rate:.0%}")
    for score in scores:
        status = "PASS" if score.passed else "FAIL"
        click.echo(f"  {status}  {score.case_id}")
    click.echo(f"Result written to {run_dir}/result.json")
    sys.exit(0 if pass_rate == 1.0 else 1)


@main.command("run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(),
    help="Path to a declarative YAML run config file.",
)
def run_config_cmd(config_path: str) -> None:
    """Load, validate, and execute a YAML run config end to end.

    Runs entirely with no webui process or HTTP call. A validation failure
    exits 2 with an actionable report and starts no agent run; a scored run
    exits 0 on pass, 1 on fail.
    """
    path = Path(config_path)
    try:
        result, verdict = execute_from_config(path)
    except ConfigValidationError as exc:
        click.secho(f"Invalid config {path}:", fg="red", err=True)
        for problem in exc.problems:
            click.echo(f"  - {problem}", err=True)
        sys.exit(2)

    click.echo(f"Run {result.run_id}: verdict {verdict}")
    for score in result.scores:
        status = "PASS" if score.passed else "FAIL"
        click.echo(f"  {status}  {score.case_id}")
    click.echo(f"Result written to {RESULTS_ROOT / result.run_id}/result.json")
    sys.exit(0 if verdict == "pass" else 1)


def _resolve_run_dir(run: str) -> Path:
    """Resolve a run identifier to its result directory.

    Accepts either a bare `run_id` (looked up under `RESULTS_ROOT`) or a
    direct path to a result directory.
    """
    direct = Path(run)
    if (direct / "result.json").exists():
        return direct
    return RESULTS_ROOT / run


def _parse_floor(raw: str) -> tuple[str, float]:
    metric, _, value = raw.partition("=")
    if not _:
        raise click.BadParameter(f"expected METRIC=FLOAT, got '{raw}'")
    try:
        return metric, float(value)
    except ValueError as exc:
        raise click.BadParameter(f"expected METRIC=FLOAT, got '{raw}'") from exc


@main.command("compare")
@click.argument("baseline_run")
@click.argument("candidate_run")
@click.option(
    "--floor",
    "floors",
    multiple=True,
    metavar="METRIC=FLOAT",
    help="Per-metric minimum the candidate run must meet or exceed. Repeatable.",
)
@click.option(
    "--allow-cross-version",
    is_flag=True,
    help="Compare even if the two runs used different dataset versions "
    "(prints a loud warning instead of refusing).",
)
@click.option(
    "--allow-regressions",
    is_flag=True,
    help="Do not fail the verdict when a previously-passing case regresses.",
)
def compare_cmd(
    baseline_run: str,
    candidate_run: str,
    floors: tuple[str, ...],
    allow_cross_version: bool,
    allow_regressions: bool,
) -> None:
    """Compare two eval runs and gate on a pass/fail verdict.

    BASELINE_RUN and CANDIDATE_RUN are each either a run id under
    `.my_coding_agent/evals/` or a direct path to a result directory.
    Exits non-zero when the verdict fails, mirroring
    `my-coding-agent-webui --check`.
    """
    baseline = load_run_result(_resolve_run_dir(baseline_run))
    candidate = load_run_result(_resolve_run_dir(candidate_run))

    if baseline.dataset != candidate.dataset and allow_cross_version:
        click.secho(
            f"WARNING: comparing different dataset versions "
            f"('{baseline.dataset}' vs '{candidate.dataset}') — "
            "results are not directly comparable.",
            fg="yellow",
            err=True,
        )

    try:
        comparison = compare_runs(
            baseline, candidate, allow_version_mismatch=allow_cross_version
        )
    except DatasetVersionMismatchError as exc:
        click.secho(f"VERDICT: FAIL — {exc}", fg="red", err=True)
        sys.exit(1)
    thresholds = ThresholdConfig(
        metric_floors=dict(_parse_floor(raw) for raw in floors),
        no_case_regressed=not allow_regressions,
    )
    verdict = evaluate_verdict(comparison, candidate, thresholds)

    click.echo(f"Baseline:  {baseline.run_id} ({baseline.dataset})")
    click.echo(f"Candidate: {candidate.run_id} ({candidate.dataset})")
    click.echo("Metric deltas:")
    for metric, delta in sorted(comparison.metric_deltas.items()):
        click.echo(f"  {metric}: {delta:+.4f}")
    if comparison.flips:
        click.echo("Flipped cases:")
        for flip in comparison.flips:
            direction = "pass -> fail" if flip.is_regression else "fail -> pass"
            click.echo(f"  {flip.case_id}: {direction}")
    else:
        click.echo("Flipped cases: none")

    if verdict.passed:
        click.secho("VERDICT: PASS", fg="green")
        sys.exit(0)
    click.secho("VERDICT: FAIL", fg="red")
    for reason in verdict.reasons:
        click.echo(f"  - {reason}")
    sys.exit(1)

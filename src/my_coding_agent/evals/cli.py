"""``my-coding-agent-eval`` console entrypoint.

Runs are produced via ``uv run my-coding-agent --config <run.yaml>`` (see
``cli.py``'s ``--config`` flag); this entrypoint only gates on the results
two of those runs left behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .compare import (
    DatasetVersionMismatchError,
    ThresholdConfig,
    compare_runs,
    evaluate_verdict,
)
from .results import RESULTS_ROOT, load_run_result


@click.group()
def main() -> None:
    """Gate eval runs on a pass/fail verdict."""


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

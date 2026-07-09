"""``my-coding-agent-eval`` console entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .cases import load_case_set
from .results import build_run_result, write_run_result
from .runner import run_case_set

_DEFAULT_CASE_DIR = Path(".my_coding_agent") / "evals" / "cases"


@click.command()
@click.option(
    "--cases",
    "case_dir",
    default=str(_DEFAULT_CASE_DIR),
    show_default=True,
    help="Directory containing one case per *.json file.",
)
def main(case_dir: str) -> None:
    """Run a case set against the agent and write a scored result record."""
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

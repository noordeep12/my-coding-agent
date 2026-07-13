"""Shared terminal rendering for a completed eval run's verdict.

One renderer for every terminal-reporting run path (the bare `--cases`
invocation and, once it lands, the `yaml-run-config` `run` subcommand) so the
full per-check verdict — pass/fail, score, rationale — is printed the same
way everywhere instead of drifting between call sites.
"""

from __future__ import annotations

import textwrap
from typing import Any

import click

from .results import EvalRunResult

_WRAP_WIDTH = 76
_DETAIL_INDENT = "      "


def _render_detail(detail: dict[str, Any]) -> list[str]:
    """Render a score's `detail` payload as indented, wrapped lines."""
    lines: list[str] = []
    criteria = detail.get("criteria")
    if isinstance(criteria, list):
        for criterion in criteria:
            name = criterion.get("name", "")
            score = criterion.get("score", "")
            lines.append(f"{_DETAIL_INDENT}{name} ({score}):")
            for wrapped in textwrap.wrap(
                str(criterion.get("rationale", "")), width=_WRAP_WIDTH
            ):
                lines.append(f"{_DETAIL_INDENT}  {wrapped}")
        overall_rationale = detail.get("overall_rationale")
        if overall_rationale:
            lines.append(f"{_DETAIL_INDENT}overall:")
            for wrapped in textwrap.wrap(str(overall_rationale), width=_WRAP_WIDTH):
                lines.append(f"{_DETAIL_INDENT}  {wrapped}")
        return lines

    for key, value in detail.items():
        for wrapped in textwrap.wrap(f"{key}: {value}", width=_WRAP_WIDTH):
            lines.append(f"{_DETAIL_INDENT}{wrapped}")
    return lines


def render_verdict(result: EvalRunResult) -> None:
    """Print the full per-check verdict, then the run-level summary.

    Prints each score's status, case id, metrics, and detail/rationale,
    followed by the existing summary line (run id, case count, pass rate).
    """
    for score in result.scores:
        status = "PASS" if score.passed else "FAIL"
        metrics = ", ".join(f"{k}={v:.2f}" for k, v in sorted(score.metrics.items()))
        header = f"  {status}  {score.case_id}"
        if metrics:
            header += f"  ({metrics})"
        click.echo(header)
        for line in _render_detail(score.detail):
            click.echo(line)

    pass_rate = result.aggregate_metrics.get("pass_rate", 0.0)
    click.echo(
        f"Run {result.run_id}: {len(result.scores)} cases, pass rate {pass_rate:.0%}"
    )

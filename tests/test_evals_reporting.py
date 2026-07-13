"""Tests for evals/reporting.py — the shared terminal verdict renderer."""

from __future__ import annotations

import click

from my_coding_agent.evals.reporting import render_verdict
from my_coding_agent.evals.results import build_run_result
from my_coding_agent.evals.schema import EvalScore


def test_judge_scored_check_s_rationale_appears_in_output(capsys):
    scores = [
        EvalScore(
            case_id="c1",
            passed=True,
            metrics={"overall_score": 0.9},
            detail={
                "criteria": [
                    {
                        "name": "clarity",
                        "score": 4,
                        "rationale": "The explanation was clear and well organized.",
                    }
                ],
                "overall_score": 0.9,
                "overall_rationale": "Solid answer overall.",
            },
        )
    ]
    result = build_run_result(
        dataset="example", scores=scores, aggregate_metrics={"pass_rate": 1.0}
    )

    with click.Context(click.Command("test")):
        render_verdict(result)

    output = capsys.readouterr().out
    assert "clarity" in output
    assert "clear and well organized" in output
    assert "Solid answer overall" in output


def test_summary_line_and_status_are_preserved(capsys):
    scores = [
        EvalScore(case_id="c1", passed=True, metrics={}, detail={}),
        EvalScore(case_id="c2", passed=False, metrics={}, detail={"reason": "nope"}),
    ]
    result = build_run_result(
        dataset="example", scores=scores, aggregate_metrics={"pass_rate": 0.5}
    )

    with click.Context(click.Command("test")):
        render_verdict(result)

    output = capsys.readouterr().out
    assert "PASS  c1" in output
    assert "FAIL  c2" in output
    assert "nope" in output
    assert f"Run {result.run_id}: 2 cases, pass rate 50%" in output

"""Tests for evals/scoring.py — the scorer contract, baseline scorer, registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from my_coding_agent.evals.schema import EvalCase
from my_coding_agent.evals.scoring import (
    ExactMatchScorer,
    RunResult,
    UnknownScorerError,
    register_scorer,
    resolve_scorer,
)


def _case(**expected):
    return EvalCase(id="c1", task="do it", scorer="exact_match", expected=expected)


def _run_result(output, errored=False):
    return RunResult(
        final_output=output,
        session_dir=Path("/tmp/session"),
        session_id="sess1",
        errored=errored,
    )


def test_baseline_scorer_passes_on_contains_match():
    score = ExactMatchScorer().score(_case(contains="pong"), _run_result("ping pong"))

    assert score.passed is True
    assert score.metrics["match"] == 1.0


def test_baseline_scorer_fails_on_no_match():
    score = ExactMatchScorer().score(_case(contains="pong"), _run_result("ping"))

    assert score.passed is False
    assert score.metrics["match"] == 0.0


def test_baseline_scorer_equals_check():
    score = ExactMatchScorer().score(_case(equals="pong"), _run_result("pong"))

    assert score.passed is True


def test_baseline_scorer_fails_run_is_scored_failed():
    score = ExactMatchScorer().score(
        _case(contains="pong"), _run_result("", errored=True)
    )

    assert score.passed is False
    assert "errored" in score.detail["reason"]


def test_unknown_scorer_ref_is_a_clear_error():
    with pytest.raises(UnknownScorerError, match="nope"):
        resolve_scorer("nope")


def test_registry_is_the_extension_point():
    class AlwaysPass:
        def score(self, case, run_result):
            from my_coding_agent.evals.schema import EvalScore

            return EvalScore(case_id=case.id, passed=True, metrics={}, detail={})

    register_scorer("always_pass", AlwaysPass())

    assert resolve_scorer("always_pass").score(_case(), _run_result("x")).passed is True

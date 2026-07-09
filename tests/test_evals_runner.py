"""Tests for evals/runner.py — the isolated, repeatable case-set runner."""

from __future__ import annotations

import os

from my_coding_agent.engine.agent import AgentNode
from my_coding_agent.evals.runner import run_case_set
from my_coding_agent.evals.schema import EvalCase


def _case(case_id, expected_contains):
    return EvalCase(
        id=case_id,
        task=f"task for {case_id}",
        scorer="exact_match",
        expected={"contains": expected_contains},
    )


def test_two_case_set_runs_both_and_isolates_workspaces(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    seen_cwds = []

    def fake_execute(self, max_steps=50):
        seen_cwds.append(os.getcwd())
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    cases = [_case("c1", "pong"), _case("c2", "pong")]
    scores, aggregate = run_case_set(cases)

    assert [score.case_id for score in scores] == ["c1", "c2"]
    assert all(score.passed for score in scores)
    assert aggregate["pass_rate"] == 1.0
    # Each case ran in its own fresh workspace, neither of which is the repo cwd.
    assert len(set(seen_cwds)) == 2
    assert str(tmp_path) not in seen_cwds
    # Runner restores the original cwd once done.
    assert os.getcwd() == str(tmp_path)


def test_erroring_agent_run_is_a_failed_case_not_a_crash(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        raise RuntimeError("boom")

    mocker.patch.object(AgentNode, "execute", fake_execute)

    cases = [_case("c1", "pong")]
    scores, aggregate = run_case_set(cases)

    assert scores[0].case_id == "c1"
    assert scores[0].passed is False
    assert aggregate["pass_rate"] == 0.0


def test_unknown_scorer_ref_is_recorded_as_failed(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    case = EvalCase(
        id="c1", task="do it", scorer="not_registered", expected={"contains": "x"}
    )
    scores, aggregate = run_case_set([case])

    assert scores[0].passed is False
    assert aggregate["pass_rate"] == 0.0


def test_empty_case_set_has_zero_pass_rate():
    scores, aggregate = run_case_set([])

    assert scores == []
    assert aggregate["pass_rate"] == 0.0

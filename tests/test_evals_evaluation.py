"""Tests for evals/evaluation.py — Evaluation/RunConfig/EvalConfig persistence + run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from my_coding_agent.pipeline.nodes.agent import AgentNode
from my_coding_agent.evals import evaluation as ev


def _dirs(tmp_path):
    return {
        "base_dir": tmp_path / "evaluations",
        "run_configs_dir": tmp_path / "run_configs",
        "eval_configs_dir": tmp_path / "eval_configs",
    }


def _make_run_config(tmp_path, **overrides):
    dirs = _dirs(tmp_path)
    data = {"name": "rc1", "model": "test-model"}
    data.update(overrides)
    return ev.create_run_config(data, base_dir=dirs["run_configs_dir"])


def _make_eval_config(tmp_path, checks=2, **overrides):
    dirs = _dirs(tmp_path)
    data = {
        "name": "ec1",
        "rules": [
            {
                "name": "rule1",
                "checks": [
                    {
                        "name": f"check{i}",
                        "evaluator": "exact_match",
                        "expected": "pong",
                        "threshold": 1.0,
                    }
                    for i in range(checks)
                ],
            }
        ],
    }
    data.update(overrides)
    return ev.create_eval_config(data, base_dir=dirs["eval_configs_dir"])


def test_run_config_round_trip_persistence(tmp_path):
    dirs = _dirs(tmp_path)
    created = _make_run_config(tmp_path, description="desc")
    loaded = ev.get_run_config(created.id, base_dir=dirs["run_configs_dir"])
    assert loaded == created


def test_eval_config_round_trip_persistence(tmp_path):
    dirs = _dirs(tmp_path)
    created = _make_eval_config(tmp_path)
    loaded = ev.get_eval_config(created.id, base_dir=dirs["eval_configs_dir"])
    assert loaded == created
    assert len(loaded.rules) == 1
    assert len(loaded.rules[0].checks) == 2


def test_minimal_run_config_is_valid(tmp_path):
    dirs = _dirs(tmp_path)
    created = ev.create_run_config(
        {"name": "minimal"}, base_dir=dirs["run_configs_dir"]
    )
    assert created.name == "minimal"
    assert created.model == ""


def test_eval_config_rejects_unknown_evaluator(tmp_path):
    dirs = _dirs(tmp_path)
    with pytest.raises(ev.UnknownEvaluatorError):
        ev.create_eval_config(
            {
                "name": "bad",
                "rules": [
                    {"name": "r1", "checks": [{"name": "c1", "evaluator": "nope"}]}
                ],
            },
            base_dir=dirs["eval_configs_dir"],
        )


def test_evaluation_round_trip_and_reuse(tmp_path):
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    eval_config = _make_eval_config(tmp_path)

    created = ev.create_evaluation(
        {
            "name": "eval1",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )
    loaded = ev.get_evaluation(created.id, base_dir=dirs["base_dir"])
    assert loaded == created
    assert loaded.run_config_id == run_config.id
    assert loaded.eval_config_id == eval_config.id

    # A second Evaluation can reference the same configs.
    created2 = ev.create_evaluation(
        {
            "name": "eval2",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )
    assert created2.run_config_id == created.run_config_id


def test_create_evaluation_rejects_unresolved_reference(tmp_path):
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    with pytest.raises(ev.UnresolvedReferenceError):
        ev.create_evaluation(
            {
                "name": "eval1",
                "run_config_id": run_config.id,
                "eval_config_id": "does-not-exist",
            },
            base_dir=dirs["base_dir"],
            run_configs_dir=dirs["run_configs_dir"],
            eval_configs_dir=dirs["eval_configs_dir"],
        )


def test_delete_evaluation_leaves_configs_intact(tmp_path):
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    eval_config = _make_eval_config(tmp_path)
    created = ev.create_evaluation(
        {
            "name": "eval1",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )

    ev.delete_evaluation(created.id, base_dir=dirs["base_dir"])

    with pytest.raises(ev.NotFoundError):
        ev.get_evaluation(created.id, base_dir=dirs["base_dir"])
    # RunConfig and EvalConfig still resolve — delete never cascades.
    assert (
        ev.get_run_config(run_config.id, base_dir=dirs["run_configs_dir"]) == run_config
    )
    assert (
        ev.get_eval_config(eval_config.id, base_dir=dirs["eval_configs_dir"])
        == eval_config
    )


def test_run_evaluation_scores_every_check(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    eval_config = _make_eval_config(tmp_path, checks=2)
    evaluation = ev.create_evaluation(
        {
            "name": "eval1",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    results_root = tmp_path / "results"
    result = ev.run_evaluation(
        evaluation,
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
        evaluations_dir=dirs["base_dir"],
        results_root=results_root,
    )

    assert len(result.scores) == 2
    assert all(score.passed for score in result.scores)
    assert result.aggregate_metrics["pass_rate"] == 1.0
    assert (results_root / result.run_id / "result.json").exists()

    updated = ev.get_evaluation(evaluation.id, base_dir=dirs["base_dir"])
    assert updated.last_run is not None
    assert updated.last_run.run_id == result.run_id
    assert updated.last_run.verdict == "pass"


def test_run_evaluation_stamps_session_id_and_verdict_is_discoverable_from_it(
    tmp_path, monkeypatch, mocker
):
    monkeypatch.chdir(tmp_path)
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    eval_config = _make_eval_config(tmp_path, checks=1)
    evaluation = ev.create_evaluation(
        {
            "name": "eval1",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    result = ev.run_evaluation(
        evaluation,
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
        evaluations_dir=dirs["base_dir"],
    )

    session_id = result.scores[0].session_id
    assert session_id is not None

    # Starting from only the session id: resolve pass/fail + rationale + record path.
    verdict_path = tmp_path / ".my_coding_agent" / session_id / "verdict.json"
    verdict = json.loads(verdict_path.read_text())
    assert verdict["run_id"] == result.run_id
    assert verdict["passed"] == result.scores[0].passed
    assert verdict["detail"] == result.scores[0].detail
    assert Path(verdict["result_path"]).exists()


def test_run_evaluation_with_no_checks_surfaces_nothing_to_score(
    tmp_path, monkeypatch, mocker
):
    monkeypatch.chdir(tmp_path)
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    eval_config = ev.create_eval_config(
        {"name": "empty", "rules": []}, base_dir=dirs["eval_configs_dir"]
    )
    evaluation = ev.create_evaluation(
        {
            "name": "eval1",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )

    def fake_execute(self, max_steps=50):
        raise AssertionError("agent should not run when there is nothing to score")

    mocker.patch.object(AgentNode, "execute", fake_execute)

    results_root = tmp_path / "results"
    result = ev.run_evaluation(
        evaluation,
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
        evaluations_dir=dirs["base_dir"],
        results_root=results_root,
    )

    assert result.scores == []
    assert result.aggregate_metrics["nothing_to_score"] == 1.0

    updated = ev.get_evaluation(evaluation.id, base_dir=dirs["base_dir"])
    assert updated.last_run.verdict == "no_checks"


def test_run_evaluation_reuses_registered_scorer(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    dirs = _dirs(tmp_path)
    run_config = _make_run_config(tmp_path)
    eval_config = _make_eval_config(tmp_path, checks=1)
    evaluation = ev.create_evaluation(
        {
            "name": "eval1",
            "run_config_id": run_config.id,
            "eval_config_id": eval_config.id,
        },
        base_dir=dirs["base_dir"],
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
    )

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)
    scorer_spy = mocker.spy(ev, "resolve_scorer")

    ev.run_evaluation(
        evaluation,
        run_configs_dir=dirs["run_configs_dir"],
        eval_configs_dir=dirs["eval_configs_dir"],
        evaluations_dir=dirs["base_dir"],
        results_root=tmp_path / "results",
    )

    scorer_spy.assert_called_with("exact_match")

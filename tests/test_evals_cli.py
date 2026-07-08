"""Tests for evals/cli.py — the `my-coding-agent-eval` console entrypoint."""

from __future__ import annotations

import json

from click.testing import CliRunner

from my_coding_agent.engine.agent import AgentNode
from my_coding_agent.evals.cli import main
from my_coding_agent.evals.results import build_run_result, write_run_result
from my_coding_agent.evals.schema import EvalScore


def _write_case(path, **fields):
    path.write_text(json.dumps(fields))


def test_cli_runs_example_set_end_to_end_and_exits_zero(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    _write_case(
        case_dir / "hello_world.json",
        id="hello_world",
        task="say pong",
        scorer="exact_match",
        expected={"contains": "pong"},
    )

    result = CliRunner().invoke(main, ["--cases", str(case_dir)])

    assert result.exit_code == 0, result.output
    assert "pass rate 100%" in result.output
    assert (tmp_path / ".my_coding_agent" / "evals").exists()


def test_cli_exits_nonzero_when_no_cases_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = CliRunner().invoke(main, ["--cases", str(empty_dir)])

    assert result.exit_code == 1


def test_compare_exits_zero_on_clean_pair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = build_run_result(
        "smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0}
    )
    candidate = build_run_result(
        "smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0}
    )
    baseline_dir = write_run_result(baseline)
    candidate_dir = write_run_result(candidate)

    result = CliRunner().invoke(
        main, ["compare", str(baseline_dir), str(candidate_dir)]
    )

    assert result.exit_code == 0, result.output
    assert "VERDICT: PASS" in result.output


def test_compare_exits_nonzero_on_regressed_pair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = build_run_result(
        "smoke@v1",
        [EvalScore("a", True, {}, {}), EvalScore("b", True, {}, {})],
        {"pass_rate": 1.0},
    )
    candidate = build_run_result(
        "smoke@v1",
        [EvalScore("a", True, {}, {}), EvalScore("b", False, {}, {})],
        {"pass_rate": 0.5},
    )
    baseline_dir = write_run_result(baseline)
    candidate_dir = write_run_result(candidate)

    result = CliRunner().invoke(
        main, ["compare", str(baseline_dir), str(candidate_dir)]
    )

    assert result.exit_code == 1
    assert "VERDICT: FAIL" in result.output
    assert "b" in result.output


def test_compare_by_run_id_under_default_results_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = build_run_result(
        "smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0}
    )
    candidate = build_run_result(
        "smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0}
    )
    write_run_result(baseline)
    write_run_result(candidate)

    result = CliRunner().invoke(
        main, ["compare", baseline.run_id, candidate.run_id]
    )

    assert result.exit_code == 0, result.output


def test_compare_refuses_cross_version_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = build_run_result(
        "smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0}
    )
    candidate = build_run_result(
        "smoke@v2", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0}
    )
    baseline_dir = write_run_result(baseline)
    candidate_dir = write_run_result(candidate)

    result = CliRunner().invoke(
        main, ["compare", str(baseline_dir), str(candidate_dir)]
    )

    assert result.exit_code != 0
    assert "not comparable" in result.output.lower()

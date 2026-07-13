"""Tests for evals/cli.py — the `my-coding-agent-eval` console entrypoint."""

from __future__ import annotations

from click.testing import CliRunner

from my_coding_agent.evals.cli import main
from my_coding_agent.evals.results import build_run_result, write_run_result
from my_coding_agent.evals.schema import EvalScore


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

    result = CliRunner().invoke(main, ["compare", baseline.run_id, candidate.run_id])

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

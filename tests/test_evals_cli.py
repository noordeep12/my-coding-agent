"""Tests for evals/cli.py — the `my-coding-agent-eval` console entrypoint."""

from __future__ import annotations

import json

from click.testing import CliRunner

from my_coding_agent.engine.agent import AgentNode
from my_coding_agent.evals.cli import main


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

"""Tests for the CLI's ``--help`` surface and the egress/exfil gate opt-outs.

Companion to ``test_cli_safety_gate_flag.py`` (which covers ``--no-safety-gate``)
and ``test_resume_cli.py`` (which covers ``--resume``/``--resume-last``
plumbing). This file locks the documented flag surface and its defaults, plus
the two remaining security-gate opt-outs' env-var wiring.
"""

from __future__ import annotations

import os

import pytest
import yaml
from click.testing import CliRunner

from my_coding_agent import cli
from my_coding_agent.engine import DEFAULT_MAX_STEPS, egress, exfil
from my_coding_agent.pipeline.nodes.agent import AgentNode


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _clean_env():
    os.environ.pop(exfil.DISABLE_ENV_VAR, None)
    os.environ.pop(egress.schema.DISABLE_ENV_VAR, None)
    yield
    os.environ.pop(exfil.DISABLE_ENV_VAR, None)
    os.environ.pop(egress.schema.DISABLE_ENV_VAR, None)


def test_help_lists_the_documented_options_with_defaults(runner):
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    for flag in (
        "--prompt",
        "--config",
        "--max-steps",
        "--resume",
        "--resume-last",
        "--no-safety-gate",
        "--no-egress-filter",
        "--no-exfil-guard",
        "--sandbox",
    ):
        assert flag in result.output

    assert "1<=x<=100" in result.output
    assert f"default: {DEFAULT_MAX_STEPS}" in result.output
    # No-value defaults surface their default behavior too (click omits
    # "[default: False]" for is_flag options since off-by-default is implicit).
    normalized_output = " ".join(result.output.split())
    assert "prompts interactively" in normalized_output
    assert "none — run a normal agent session instead" in normalized_output


def test_config_flag_exits_zero_on_pass(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"task": "say pong"},
                "evaluation": {
                    "checks": [{"evaluator": "exact_match", "expected": "pong"}]
                },
            }
        )
    )

    result = CliRunner().invoke(cli.main, ["--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "verdict pass" in result.output


def test_config_flag_exits_two_on_invalid_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump({"bogus": {}}))

    result = CliRunner().invoke(cli.main, ["--config", str(config_path)])

    assert result.exit_code == 2
    assert "Invalid config" in result.output
    assert not (tmp_path / ".my_coding_agent").exists()


def test_config_flag_rejects_resume_combination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump({"run": {"task": "say pong"}}))

    result = CliRunner().invoke(
        cli.main, ["--config", str(config_path), "--resume", "abc123"]
    )

    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_no_egress_filter_flag_sets_env_var(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    mocker.patch.object(cli, "_build_fresh_agent", return_value=fake_agent)
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli.main, ["--prompt", "do a thing", "--no-egress-filter"]
        )
        assert result.exit_code == 0
        assert os.environ.get(egress.schema.DISABLE_ENV_VAR) == "1"


def test_no_exfil_guard_flag_sets_env_var(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    mocker.patch.object(cli, "_build_fresh_agent", return_value=fake_agent)
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--prompt", "do a thing", "--no-exfil-guard"])
        assert result.exit_code == 0
        assert os.environ.get(exfil.DISABLE_ENV_VAR) == "1"

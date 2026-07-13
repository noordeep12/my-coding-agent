"""Tests for the ``--no-safety-gate`` CLI flag (issue #124 follow-up).

The flag is a thin opt-out: it sets ``MCA_DISABLE_DANGEROUS_COMMAND_GATE`` for
the process before the agent runs, so ``policy.evaluate`` sees it on every
subsequent call. No plumbing through ``AgentNode``/``ToolExecutor`` is needed.
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from my_coding_agent import cli
from my_coding_agent.engine.tool_execution import policy


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _clean_env():
    os.environ.pop(policy.DISABLE_ENV_VAR, None)
    yield
    os.environ.pop(policy.DISABLE_ENV_VAR, None)


def test_no_safety_gate_flag_sets_env_var_and_warns(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    mocker.patch.object(cli, "_build_fresh_agent", return_value=fake_agent)
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--prompt", "do a thing", "--no-safety-gate"])
        assert result.exit_code == 0
        assert os.environ.get(policy.DISABLE_ENV_VAR) == "1"
        assert "refusal gate is off" in result.output.lower()
        assert policy.is_gate_disabled() is True


def test_without_the_flag_gate_stays_enabled(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    mocker.patch.object(cli, "_build_fresh_agent", return_value=fake_agent)
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--prompt", "do a thing"])
        assert result.exit_code == 0
        assert policy.DISABLE_ENV_VAR not in os.environ
        assert policy.is_gate_disabled() is False

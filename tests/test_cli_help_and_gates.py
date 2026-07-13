"""Tests for the CLI's ``--help`` surface and the egress/exfil gate opt-outs.

Companion to ``test_cli_safety_gate_flag.py`` (which covers ``--no-safety-gate``)
and ``test_resume_cli.py`` (which covers ``--resume``/``--resume-last``
plumbing). This file locks the documented nine-option flag surface and its
defaults, plus the two remaining security-gate opt-outs' env-var wiring.
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from my_coding_agent import cli
from my_coding_agent.engine import DEFAULT_MAX_STEPS, egress, exfil


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


def test_help_lists_the_nine_documented_options(runner):
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    for flag in (
        "--prompt",
        "--interactive",
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

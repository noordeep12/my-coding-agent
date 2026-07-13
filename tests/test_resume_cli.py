"""Tests for the CLI resume surface (run-resilience D5/D6).

Drive the Click command with ``CliRunner`` in an isolated filesystem. ``AgentNode``
is patched so no real agent (network, session log) is built — the tests assert the
CLI's own behavior: it loads the right checkpoint, refuses cleanly on a bad id, and
exits non-zero with a resume hint when a run fails.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from my_coding_agent.engine.checkpoint import Checkpoint, save_checkpoint
from my_coding_agent.engine.llm.errors import LLMTransportError
from my_coding_agent import cli


@pytest.fixture
def runner():
    return CliRunner()


def _write_checkpoint(session_id="deadbeef1234", step=4):
    cp = Checkpoint(
        session_id=session_id,
        step_num=step,
        last_prompt_tokens=99,
        messages=[{"role": "user", "content": "resume me"}],
    )
    save_checkpoint(Path(".my_coding_agent") / session_id, cp)
    return cp


def test_resume_invalid_id_refuses_and_touches_nothing(runner, mocker):
    built = mocker.patch.object(cli.AgentNode, "from_checkpoint")
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--resume", "no-such-session"])
        assert result.exit_code == 2
        assert "Cannot resume" in result.output
        # Nothing built, nothing created.
        built.assert_not_called()
        assert not Path(".my_coding_agent").exists()


def test_resume_last_with_no_checkpoints_refuses(runner, mocker):
    built = mocker.patch.object(cli.AgentNode, "from_checkpoint")
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--resume-last"])
        assert result.exit_code == 2
        assert "No resumable session" in result.output
        built.assert_not_called()


def test_resume_loads_checkpoint_and_runs(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    from_cp = mocker.patch.object(
        cli.AgentNode, "from_checkpoint", return_value=fake_agent
    )
    with runner.isolated_filesystem():
        cp = _write_checkpoint()
        result = runner.invoke(
            cli.main, ["--resume", "deadbeef1234", "--max-steps", "7"]
        )
        assert result.exit_code == 0
        # The right checkpoint was loaded and handed to from_checkpoint.
        loaded = from_cp.call_args.args[0]
        assert loaded.session_id == cp.session_id
        assert loaded.step_num == cp.step_num
        fake_agent.execute.assert_called_once_with(max_steps=7)


def test_resume_last_selects_newest(runner, mocker):
    import os
    import time

    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    from_cp = mocker.patch.object(
        cli.AgentNode, "from_checkpoint", return_value=fake_agent
    )
    with runner.isolated_filesystem():
        _write_checkpoint(session_id="older", step=1)
        _write_checkpoint(session_id="newer", step=2)
        now = time.time()
        os.utime(
            Path(".my_coding_agent") / "older" / "checkpoint.json",
            (now - 100, now - 100),
        )
        os.utime(Path(".my_coding_agent") / "newer" / "checkpoint.json", (now, now))
        result = runner.invoke(cli.main, ["--resume-last"])
        assert result.exit_code == 0
        assert from_cp.call_args.args[0].session_id == "newer"


def test_run_failure_exits_nonzero_with_resume_hint(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.session_id = "abc123"
    fake_agent.failure_session_id = None
    fake_agent.failure_error = LLMTransportError("down")
    mocker.patch.object(cli, "AgentNode")
    mocker.patch.object(cli, "_build_fresh_agent", return_value=fake_agent)
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--prompt", "do a thing"])
        assert result.exit_code == 1
        assert "--resume abc123" in result.output
        assert "unrecoverable LLM failure" in result.output


def test_healthy_run_exits_zero(runner, mocker):
    fake_agent = mocker.Mock()
    fake_agent.failure_error = None
    mocker.patch.object(cli, "_build_fresh_agent", return_value=fake_agent)
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["--prompt", "do a thing"])
        assert result.exit_code == 0
        fake_agent.execute.assert_called_once()

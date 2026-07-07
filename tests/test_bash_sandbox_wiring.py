"""Sandbox wiring into ``ToolRegistry.bash`` (issue #25, bash-os-sandbox change).

Covers: flag off is byte-identical to today's bash behavior with no sandbox
events; flag on wraps the command via `sandbox-exec` and denies real
out-of-scope writes / network end-to-end; unsupported-host degradation is an
explicit `ok:false` refusal, never a silent unconfined execution.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from my_coding_agent.engine import sandbox
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry


@pytest.fixture(autouse=True)
def _sandbox_off_by_default(monkeypatch):
    monkeypatch.delenv(sandbox.ENV_VAR, raising=False)


class TestFlagOffIsUnchanged:
    def test_no_sandbox_wrapping_when_flag_off(self, mocker):
        run = mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args="echo hi", returncode=0, stdout="hi\n", stderr=""
            ),
        )
        out = ToolsRegistry().bash("echo hi")
        parsed = json.loads(out)
        assert parsed == {"stdout": "hi", "stderr": "", "exit_code": 0, "ok": True}
        called_args, called_kwargs = run.call_args
        assert called_args[0] == "echo hi"
        assert called_kwargs["shell"] is True

    def test_no_sandbox_denial_event_when_flag_off(self, mocker):
        mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args="false", returncode=1, stdout="", stderr="Operation not permitted"
            ),
        )
        recorder = mocker.Mock()
        mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.current_recorder"
        ).get.return_value = recorder
        ToolsRegistry().bash("false")
        recorder.record_sandbox_denial.assert_not_called()


class TestFlagOnWrapsCommand:
    def test_wraps_via_sandbox_exec_when_enabled(self, monkeypatch, mocker, tmp_path):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        mocker.patch(
            "my_coding_agent.engine.sandbox.probe_host_capability",
            return_value=sandbox.HostCapability(supported=True, reason=""),
        )
        run = mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="hi\n", stderr=""
            ),
        )
        out = ToolsRegistry(base_dir=str(tmp_path)).bash("echo hi")
        parsed = json.loads(out)
        assert parsed["ok"] is True
        called_args, called_kwargs = run.call_args
        assert called_args[0][0] == "sandbox-exec"
        assert called_kwargs["shell"] is False

    def test_denial_event_recorded_when_sandboxed_command_denied(
        self, monkeypatch, mocker, tmp_path
    ):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        mocker.patch(
            "my_coding_agent.engine.sandbox.probe_host_capability",
            return_value=sandbox.HostCapability(supported=True, reason=""),
        )
        mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="touch: /etc/foo: Operation not permitted",
            ),
        )
        recorder = mocker.Mock()
        mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.current_recorder"
        ).get.return_value = recorder
        ToolsRegistry(base_dir=str(tmp_path), step_num=3).bash("touch /etc/foo")
        recorder.record_sandbox_denial.assert_called_once()
        _, kwargs = recorder.record_sandbox_denial.call_args
        assert kwargs["command"] == "touch /etc/foo"
        assert kwargs["exit_code"] == 1
        assert kwargs["step"] == 3

    def test_ordinary_failure_is_not_recorded_as_denial(
        self, monkeypatch, mocker, tmp_path
    ):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        mocker.patch(
            "my_coding_agent.engine.sandbox.probe_host_capability",
            return_value=sandbox.HostCapability(supported=True, reason=""),
        )
        mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="no such file"
            ),
        )
        recorder = mocker.Mock()
        mocker.patch(
            "my_coding_agent.engine.tool_registry.registry.current_recorder"
        ).get.return_value = recorder
        ToolsRegistry(base_dir=str(tmp_path)).bash("cat missing")
        recorder.record_sandbox_denial.assert_not_called()


class TestUnsupportedHostDegradation:
    def test_enabling_on_unsupported_host_refuses_without_running(
        self, monkeypatch, mocker, tmp_path
    ):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        mocker.patch(
            "my_coding_agent.engine.sandbox.probe_host_capability",
            return_value=sandbox.HostCapability(
                supported=False, reason="unsupported platform 'Linux'"
            ),
        )
        run = mocker.patch("my_coding_agent.engine.tool_registry.registry.subprocess.run")
        out = ToolsRegistry(base_dir=str(tmp_path)).bash("echo hi")
        parsed = json.loads(out)
        assert parsed["ok"] is False
        assert "unavailable on this host" in parsed["stderr"]
        run.assert_not_called()


@pytest.mark.skipif(
    platform.system() != "Darwin" or shutil.which("sandbox-exec") is None,
    reason="requires macOS with sandbox-exec",
)
class TestEndToEndDenialAndSuccess:
    def test_out_of_scope_write_denied_end_to_end(self, monkeypatch, tmp_path):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Outside the workspace, /tmp, /private/tmp, AND $TMPDIR (which
        # tmp_path is always nested under) — a genuine out-of-scope target.
        outside = Path("/private/var/tmp") / f"mca_sandbox_e2e_{uuid.uuid4().hex}.txt"
        try:
            parsed = json.loads(
                ToolsRegistry(base_dir=str(workspace)).bash(f"echo hi > {outside}")
            )
            assert parsed["ok"] is False
            assert not outside.exists()
        finally:
            outside.unlink(missing_ok=True)

    def test_in_scope_write_succeeds_end_to_end(self, monkeypatch, tmp_path):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "file.txt"
        parsed = json.loads(
            ToolsRegistry(base_dir=str(workspace)).bash(f"echo hi > {target}")
        )
        assert parsed["ok"] is True
        assert target.read_text().strip() == "hi"

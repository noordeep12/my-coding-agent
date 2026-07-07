"""Unit tests for engine.sandbox — profile generation, wrapping, capability probe
(issue #25, bash-os-sandbox change).

Covers: representative out-of-scope write is denied, in-scope write is
allowed, a blind network call is denied, and whole-process-tree coverage (a
spawned child inherits the same confinement) — end to end, invoking the real
``sandbox-exec`` binary. Skipped entirely off macOS or without the binary.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from my_coding_agent.engine import sandbox

# A location outside the workspace root, `/tmp`, `/private/tmp`, AND the
# process's own `$TMPDIR` (which pytest's `tmp_path` fixture is always nested
# under) — the only macOS system temp dir left that default_scope's allowlist
# does not cover, so it is a genuine out-of-scope target.
_OUTSIDE_ROOT = Path("/private/var/tmp")

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or shutil.which("sandbox-exec") is None,
    reason="requires macOS with sandbox-exec",
)


def _run(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=10)


class TestHostCapability:
    def test_supported_on_this_host(self):
        capability = sandbox.probe_host_capability()
        assert capability.supported is True


class TestIsEnabled:
    def test_defaults_off(self, monkeypatch):
        monkeypatch.delenv(sandbox.ENV_VAR, raising=False)
        assert sandbox.is_enabled() is False

    def test_on_when_set_truthy(self, monkeypatch):
        monkeypatch.setenv(sandbox.ENV_VAR, "1")
        assert sandbox.is_enabled() is True

    def test_off_when_set_to_zero_or_false(self, monkeypatch):
        monkeypatch.setenv(sandbox.ENV_VAR, "0")
        assert sandbox.is_enabled() is False
        monkeypatch.setenv(sandbox.ENV_VAR, "false")
        assert sandbox.is_enabled() is False


class TestOutOfScopeWriteDenied:
    def test_write_outside_workspace_is_denied(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = _OUTSIDE_ROOT / f"mca-sandbox-test-{uuid.uuid4().hex}.txt"
        scope = sandbox.default_scope(workspace)
        try:
            argv = sandbox.wrap_command(f"echo hi > {outside}", scope)
            result = _run(argv)
            assert result.returncode != 0
            assert not outside.exists()
        finally:
            outside.unlink(missing_ok=True)


class TestInScopeWriteAllowed:
    def test_write_inside_workspace_succeeds(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "inside.txt"
        scope = sandbox.default_scope(workspace)
        argv = sandbox.wrap_command(f"echo hi > {target}", scope)
        result = _run(argv)
        assert result.returncode == 0
        assert target.read_text().strip() == "hi"


class TestNetworkDenied:
    def test_blind_network_call_is_denied(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scope = sandbox.default_scope(workspace)
        argv = sandbox.wrap_command("curl -sS --max-time 3 http://example.com", scope)
        result = _run(argv)
        assert result.returncode != 0


class TestProcessTreeCoverage:
    def test_spawned_child_is_also_confined(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = _OUTSIDE_ROOT / f"mca-sandbox-test-child-{uuid.uuid4().hex}.txt"
        scope = sandbox.default_scope(workspace)
        try:
            # The child (`sh -c ...`) attempts the out-of-scope write, not the
            # top-level command itself — proves confinement covers descendants.
            argv = sandbox.wrap_command(f"sh -c 'echo hi > {outside}'", scope)
            result = _run(argv)
            assert result.returncode != 0
            assert not outside.exists()
        finally:
            outside.unlink(missing_ok=True)


class TestIsLikelyDenial:
    def test_matches_operation_not_permitted(self):
        assert sandbox.is_likely_denial("bash: foo: Operation not permitted")

    def test_does_not_match_unrelated_failure(self):
        assert not sandbox.is_likely_denial("grep: no matches found")

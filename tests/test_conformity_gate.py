"""Tests for the pre-commit conformity gate (.hooks/check_conformity_report.py)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / ".hooks" / "check_conformity_report.py"
STATE_FILE = ".hooks/.conformity_state.json"


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / ".hooks").mkdir(parents=True)
    _run(["git", "init", "-q"], r)
    _run(["git", "config", "user.email", "t@t"], r)
    _run(["git", "config", "user.name", "t"], r)
    (r / "src" / "x.py").write_text("a = 1\n")
    _run(["git", "add", "."], r)
    _run(["git", "commit", "-qm", "init"], r)
    return r


def _diff_hash(repo):
    out = _run(["git", "diff", "HEAD", "--", "src/"], repo).stdout
    return hashlib.sha256(out.encode()).hexdigest() if out.strip() else ""


def _gate(repo):
    return _run([sys.executable, str(GATE)], repo)


def _write_state(repo, diff_hash, status):
    state = {
        "diff_hash": diff_hash,
        "status": status,
        "mode": None,
        "auto_iterations": 0,
        "dispositioned_gaps": [],
    }
    (repo / STATE_FILE).write_text(json.dumps(state))


def test_no_code_change_passes(repo):
    assert _gate(repo).returncode == 0


def test_missing_state_blocks(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    assert _gate(repo).returncode == 1


def test_stale_state_blocks(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_state(repo, "deadbeef", "resolved")
    assert _gate(repo).returncode == 1


def test_pending_state_blocks(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_state(repo, _diff_hash(repo), "pending")
    assert _gate(repo).returncode == 1


def test_resolved_state_allows(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_state(repo, _diff_hash(repo), "resolved")
    assert _gate(repo).returncode == 0

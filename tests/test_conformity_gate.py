"""Tests for the pre-commit conformity gate (.hooks/check_conformity_report.py)."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / ".hooks" / "check_conformity_report.py"


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
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


def _write_report(repo, diff_hash, state):
    (repo / "conformity.md").write_text(
        "# Conformity Report\n"
        f"<!-- conformity-meta\ndiff_hash: {diff_hash}\nstate: {state}\n-->\n"
    )


def test_no_code_change_passes(repo):
    assert _gate(repo).returncode == 0


def test_missing_report_blocks(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    assert _gate(repo).returncode == 1


def test_stale_report_blocks(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_report(repo, "deadbeef", "approved")
    assert _gate(repo).returncode == 1


def test_blocked_state_blocks(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_report(repo, _diff_hash(repo), "blocked")
    assert _gate(repo).returncode == 1


def test_pass_state_allows(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_report(repo, _diff_hash(repo), "pass")
    assert _gate(repo).returncode == 0


def test_approved_state_allows(repo):
    (repo / "src" / "x.py").write_text("a = 2\n")
    _write_report(repo, _diff_hash(repo), "approved")
    assert _gate(repo).returncode == 0

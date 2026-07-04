"""Tests for the `my-coding-agent-traces --check` sum-check CLI surface (D4)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from my_coding_agent.viewer.server import _cli


def _write_session(base_dir, session_id, by_kind, calls, grand_total, report_source):
    sdir = base_dir / session_id
    sdir.mkdir(parents=True)
    data = {
        "session_id": session_id,
        "llm_calls": calls,
        "rollup": {"by_kind": by_kind, "descendants": [], "grand_total": grand_total},
    }
    (sdir / "session_data.json").write_text(json.dumps(data))
    (sdir / "events.jsonl").write_text(
        json.dumps({"type": "report", "source": report_source}) + "\n"
    )


def test_check_passes_with_exit_zero(tmp_path):
    tokens = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    _write_session(
        tmp_path,
        "s1",
        {"main": tokens},
        [{"kind": "main", "prompt": 10, "completion": 5, "total": 15}],
        tokens,
        "verbatim",
    )
    runner = CliRunner()
    result = runner.invoke(_cli, ["--dir", str(tmp_path), "--check", "s1"])
    assert result.exit_code == 0
    assert "PASS s1" in result.output


def test_check_fails_with_exit_one_naming_kind(tmp_path):
    bad = {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 999}
    _write_session(
        tmp_path,
        "s2",
        {"main": bad},
        [{"kind": "main", "prompt": 10, "completion": 5, "total": 15}],
        bad,
        "verbatim",
    )
    runner = CliRunner()
    result = runner.invoke(_cli, ["--dir", str(tmp_path), "--check", "s2"])
    assert result.exit_code == 1
    assert "FAIL s2" in result.output
    assert "main" in result.output


def test_check_unverifiable_for_missing_session(tmp_path):
    runner = CliRunner()
    result = runner.invoke(_cli, ["--dir", str(tmp_path), "--check", "ghost"])
    assert result.exit_code == 0
    assert "UNVERIFIABLE ghost" in result.output

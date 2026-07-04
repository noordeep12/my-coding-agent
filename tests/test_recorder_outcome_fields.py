"""Tests for tool_call events' top-level outcome fields (ok/error/error_class).

See openspec/changes/tool-call-outcome-fields (spec: tool-call-outcome-capture).
"""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import Recorder
from my_coding_agent.pipeline.anomaly import error_signature


def _make_recorder(tmp_path):
    sid = "aabbccdd"
    sdir = tmp_path / sid
    sdir.mkdir()
    return Recorder(session_id=sid, session_dir=sdir), sdir / "events.jsonl"


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _emit_tool_call(rec, name, args, result, ok, error):
    rec.before_tool(name, args)
    rec.after_tool(name, args, result, ok, error)


class TestSuccessfulCall:
    def test_ok_true_no_error_fields(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(rec, "bash", {"command": "ls"}, "result", True, None)
        ev = _read_events(path)[-1]
        assert ev["ok"] is True
        assert "error" not in ev
        assert "error_class" not in ev


class TestToolReportedFailure:
    def test_ok_false_carries_error_and_class(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(
            rec,
            "bash",
            {"command": "cat missing"},
            '{"ok": false}',
            False,
            "Error: no such file or directory",
        )
        ev = _read_events(path)[-1]
        assert ev["ok"] is False
        assert ev["error"] == "Error: no such file or directory"
        assert ev["error_class"]


class TestExecutorHandledFailures:
    def test_not_found(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(
            rec,
            "nope",
            {},
            "result",
            False,
            "Error: tool 'nope' not found. Available tools: []",
        )
        ev = _read_events(path)[-1]
        assert ev["ok"] is False
        assert "not found" in ev["error"]

    def test_wrong_args(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(
            rec,
            "bash",
            {},
            "result",
            False,
            "Error: wrong arguments for 'bash': missing 1 required positional argument",
        )
        ev = _read_events(path)[-1]
        assert ev["ok"] is False
        assert "wrong arguments" in ev["error"]

    def test_raised_recoverable_exception(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(
            rec,
            "read_file",
            {"path": "x"},
            "result",
            False,
            "Error: tool 'read_file' raised FileNotFoundError: x",
        )
        ev = _read_events(path)[-1]
        assert ev["ok"] is False
        assert ev["error_class"] == "FileNotFoundError"


class TestErrorClassification:
    def test_traceback_text_yields_exception_token(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        error_text = (
            'Traceback (most recent call last):\n  File "x.py"\n'
            "json.decoder.JSONDecodeError: Expecting value"
        )
        _emit_tool_call(rec, "bash", {}, "result", False, error_text)
        ev = _read_events(path)[-1]
        assert ev["error_class"] == "json.decoder.JSONDecodeError"

    def test_non_exception_text_yields_digit_stripped_first_line(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(
            rec, "bash", {}, "result", False, "Error: file not found at line 42"
        )
        ev = _read_events(path)[-1]
        assert ev["error_class"] == "Error: file not found at line"


class TestClassAgreesWithAnomalyDetector:
    def test_recorder_error_class_matches_anomaly_signature(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        error_text = "Error: tool 'bash' raised ValueError: bad value"
        _emit_tool_call(rec, "bash", {"command": "x"}, "result", False, error_text)
        ev = _read_events(path)[-1]
        signature = error_signature({"name": "bash", "error": error_text})
        assert signature == f"bash|{ev['error_class']}"


class TestOnePassFailureQuery:
    def test_single_parse_per_line_lists_failures_with_class(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        _emit_tool_call(rec, "bash", {"command": "ok"}, "result", True, None)
        _emit_tool_call(rec, "bash", {"command": "bad"}, "result", False, "Error: boom")
        failures = [
            json.loads(line)["error_class"]
            for line in path.read_text().splitlines()
            if line.strip()
            and json.loads(line).get("type") == "tool_call"
            and json.loads(line).get("ok") is False
        ]
        assert failures == ["Error: boom"]

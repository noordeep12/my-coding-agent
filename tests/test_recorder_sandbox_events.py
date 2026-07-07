"""Regression tests for Recorder.record_sandbox_activation / record_sandbox_denial
(issue #25, bash-os-sandbox change).

Locks in: activation + denial rows are answerable with one ``json.loads``,
sandbox-off runs emit none of either (pre-change traces load unchanged).
"""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import (
    SANDBOX_ACTIVATION,
    SANDBOX_DENIAL,
    Recorder,
)


def _make_recorder(tmp_path):
    sid = "aabbccdd"
    sdir = tmp_path / sid
    sdir.mkdir()
    return Recorder(session_id=sid, session_dir=sdir), sdir / "events.jsonl"


def _read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRecordSandboxActivation:
    def test_emits_correct_type_and_fields(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_sandbox_activation(
            workspace_root="/Users/dev/project",
            extra_write_paths=["/tmp", "/private/tmp"],
        )
        ev = _read_events(path)[-1]
        assert ev["type"] == SANDBOX_ACTIVATION
        assert ev["workspace_root"] == "/Users/dev/project"
        assert ev["extra_write_paths"] == ["/tmp", "/private/tmp"]
        assert "started_at" in ev

    def test_one_json_loads_answers_activation(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_sandbox_activation(workspace_root="/x", extra_write_paths=[])
        line = path.read_text().splitlines()[-1]
        ev = json.loads(line)
        assert ev["type"] == SANDBOX_ACTIVATION


class TestRecordSandboxDenial:
    def test_emits_correct_type_and_fields(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_sandbox_denial(
            command="touch /etc/foo",
            exit_code=1,
            stderr="Operation not permitted",
            step=2,
        )
        ev = _read_events(path)[-1]
        assert ev["type"] == SANDBOX_DENIAL
        assert ev["command"] == "touch /etc/foo"
        assert ev["exit_code"] == 1
        assert ev["stderr"] == "Operation not permitted"
        assert ev["step"] == 2


class TestSandboxOffEmitsNoEvents:
    def test_no_sandbox_rows_when_never_recorded(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.start(label="Main Agent", model="test-model", context_window=8192)
        rec.finish(stop_reason="clean", steps=1, elapsed_s=0.1)
        events = _read_events(path)
        assert not any(
            e["type"] in (SANDBOX_ACTIVATION, SANDBOX_DENIAL) for e in events
        )

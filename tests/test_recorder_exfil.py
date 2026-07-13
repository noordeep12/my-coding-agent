"""Regression tests for Recorder.record_exfil (issue #127).

Locks in: a block row is answerable with one ``json.loads`` and carries only
the matched category (never a secret value), and a run with the guard
disabled emits no exfil rows at all.
"""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import EXFIL, Recorder


def _make_recorder(tmp_path):
    sid = "aabbccdd"
    sdir = tmp_path / sid
    sdir.mkdir()
    return Recorder(session_id=sid, session_dir=sdir), sdir / "events.jsonl"


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRecordExfil:
    def test_emits_correct_type(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_exfil(tool_name="fetch_web", category="ssh_key", step=1)
        events = _read_events(path)
        assert events[0]["type"] == EXFIL

    def test_row_answerable_with_one_json_loads_and_has_no_secret_value(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_exfil(tool_name="fetch_web", category="ssh_key", step=2)
        raw_line = path.read_text().splitlines()[0]
        event = json.loads(raw_line)
        assert event["tool_name"] == "fetch_web"
        assert event["category"] == "ssh_key"
        assert event["step"] == 2
        assert "started_at" in event
        assert "id_ed25519" not in raw_line
        assert "-----BEGIN" not in raw_line

    def test_no_events_when_guard_never_triggers(self, tmp_path):
        _rec, path = _make_recorder(tmp_path)
        # A run where the guard never fires writes no file at all (no
        # record_exfil call made) — session directory stays untouched.
        assert not path.exists()

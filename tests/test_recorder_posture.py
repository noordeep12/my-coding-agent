"""Protection-posture recording on session_start (issue #130).

The posture (``sandboxed`` vs ``screened_only``) is recorded once per run on
the ``session_start`` row, passively — recording changes no enforcement — and
must be readable with a single ``json.loads`` per line, like every other
event.
"""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import Recorder


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_session_start_records_screened_only_posture(tmp_path):
    sid = "posture01"
    sdir = tmp_path / sid
    sdir.mkdir()
    rec = Recorder(session_id=sid, session_dir=sdir)

    rec.start("Agent", "test-model", 8192, "screened_only")

    events = _read_events(sdir / "events.jsonl")
    assert events[0]["posture"] == "screened_only"


def test_session_start_records_sandboxed_posture(tmp_path):
    sid = "posture02"
    sdir = tmp_path / sid
    sdir.mkdir()
    rec = Recorder(session_id=sid, session_dir=sdir)

    rec.start("Agent", "test-model", 8192, "sandboxed")

    events = _read_events(sdir / "events.jsonl")
    assert events[0]["posture"] == "sandboxed"

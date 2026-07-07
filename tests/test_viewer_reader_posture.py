"""Tests for the protection-posture viewer surfacing (issue #130).

The ``session_start`` event's ``posture`` field (``sandboxed`` or
``screened_only``) is read straight onto ``TraceSession.posture``. Traces
recorded before posture capture existed carry no ``posture`` key at all and
must render unchanged (``posture`` stays ``None``, no indicator).
"""

import json
from pathlib import Path

from my_coding_agent.viewer.reader import load_session


def _ev(type: str, **kw):
    return {"type": type, **kw}


def _write_events(path: Path, events: list) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def _session_events(session_id: str, *, posture: str | None) -> list:
    start_kwargs = dict(
        session_id=session_id,
        label="Test",
        model="gpt-4o-mini",
        context_window=8192,
        started_at="2026-01-01T10:00:00",
        parent_session_id=None,
    )
    if posture is not None:
        start_kwargs["posture"] = posture
    return [
        _ev("session_start", **start_kwargs),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=0,
            elapsed_s=0.1,
            ended_at="2026-01-01T10:00:01",
        ),
    ]


def _load(tmp_path, sid, events):
    sdir = tmp_path / sid
    sdir.mkdir()
    ep = sdir / "events.jsonl"
    _write_events(ep, events)
    return load_session(ep)


class TestPostureSurfacing:
    def test_screened_only_session_is_marked(self, tmp_path):
        sid = "posture0001"
        session = _load(tmp_path, sid, _session_events(sid, posture="screened_only"))
        assert session.posture == "screened_only"

    def test_sandboxed_session_reflects_enforcement(self, tmp_path):
        sid = "posture0002"
        session = _load(tmp_path, sid, _session_events(sid, posture="sandboxed"))
        assert session.posture == "sandboxed"

    def test_pre_change_trace_has_no_posture(self, tmp_path):
        sid = "posture0003"
        session = _load(tmp_path, sid, _session_events(sid, posture=None))
        assert session.posture is None

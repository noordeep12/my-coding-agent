"""Tests that the reader side-reads a session's `verdict.json` sibling into
`TraceSession.verdict`, and degrades to `None` on absence or malformed content
(session-verdict-visibility)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from my_coding_agent.viewer.reader import load_session


def _ev(type_: str, **kw):
    return {"type": type_, **kw}


def _write_events(path: Path, events: list) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def _minimal_events(session_id: str = "verdictsession001") -> list:
    return [
        _ev(
            "session_start",
            session_id=session_id,
            label="Test",
            model="gpt-4o-mini",
            context_window=8192,
            started_at="2026-01-01T10:00:00",
        ),
        _ev(
            "llm_call",
            call=1,
            kind="main",
            latency_s=1.0,
            prompt=100,
            completion=50,
            total=150,
            context_window=8192,
            messages=None,
            response={"content": "ok", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="2026-01-01T10:00:01",
        ),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=1,
            elapsed_s=1.0,
            ended_at="2026-01-01T10:00:02",
        ),
    ]


_VERDICT = {
    "run_id": "run-123",
    "case_id": "case-1",
    "passed": True,
    "metrics": {"score": 0.9},
    "detail": "looks good",
    "result_path": "/tmp/result.json",
}


class TestWellFormedVerdict:
    def test_verdict_round_trips_into_session(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _minimal_events())
        (tmp_path / "verdict.json").write_text(json.dumps(_VERDICT))
        session = load_session(events_path)
        assert session.verdict == _VERDICT


class TestAbsentVerdict:
    def test_missing_verdict_json_yields_none(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _minimal_events())
        session = load_session(events_path)
        assert session.verdict is None


class TestMalformedVerdict:
    def test_bad_json_yields_none_and_logs_warning(self, tmp_path, caplog):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _minimal_events())
        (tmp_path / "verdict.json").write_text("{not valid json")
        with caplog.at_level(logging.WARNING):
            session = load_session(events_path)
        assert session.verdict is None
        assert any("verdict" in rec.message for rec in caplog.records)

    def test_non_object_json_yields_none_and_logs_warning(self, tmp_path, caplog):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _minimal_events())
        (tmp_path / "verdict.json").write_text(json.dumps([1, 2, 3]))
        with caplog.at_level(logging.WARNING):
            session = load_session(events_path)
        assert session.verdict is None
        assert any("verdict" in rec.message for rec in caplog.records)

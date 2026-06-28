"""Regression tests for Recorder.record_token_tracking and record_finish_check."""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import (
    FINISH_CHECK,
    TOKEN_TRACKING,
    Recorder,
)


def _make_recorder(tmp_path):
    sid = "aabbccdd"
    sdir = tmp_path / sid
    sdir.mkdir()
    return Recorder(session_id=sid, session_dir=sdir), sdir / "events.jsonl"


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRecordTokenTracking:
    def test_emits_correct_type(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_token_tracking(
            step=1,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ctx_pct=1.23456,
            context_window=8192,
        )
        events = _read_events(path)
        assert events[-1]["type"] == TOKEN_TRACKING

    def test_fields_present(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_token_tracking(
            step=2,
            prompt_tokens=200,
            completion_tokens=80,
            total_tokens=280,
            ctx_pct=2.5,
            context_window=16384,
        )
        ev = _read_events(path)[-1]
        assert ev["step"] == 2
        assert ev["prompt_tokens"] == 200
        assert ev["completion_tokens"] == 80
        assert ev["total_tokens"] == 280
        assert ev["context_window"] == 16384

    def test_ctx_pct_is_rounded_to_one_decimal(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_token_tracking(
            step=1,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ctx_pct=3.14159,
            context_window=8192,
        )
        ev = _read_events(path)[-1]
        assert ev["ctx_pct"] == 3.1

    def test_started_at_is_present(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_token_tracking(
            step=1,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ctx_pct=0.0,
            context_window=8192,
        )
        ev = _read_events(path)[-1]
        assert "started_at" in ev and ev["started_at"]


class TestRecordFinishCheck:
    def test_emits_correct_type(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_finish_check(step=1, finish_reason="stop", signal="STOP")
        events = _read_events(path)
        assert events[-1]["type"] == FINISH_CHECK

    def test_fields_present(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_finish_check(step=3, finish_reason="tool_use", signal="CONTINUE")
        ev = _read_events(path)[-1]
        assert ev["step"] == 3
        assert ev["finish_reason"] == "tool_use"
        assert ev["signal"] == "CONTINUE"

    def test_started_at_is_present(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_finish_check(step=1, finish_reason="stop", signal="STOP")
        ev = _read_events(path)[-1]
        assert "started_at" in ev and ev["started_at"]

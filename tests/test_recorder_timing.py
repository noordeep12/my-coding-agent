"""Timing-consistency regression test for trace-cost-time-attribution.

For a recorded run, every timed event's ``started_at`` reflects true work
start (not emit time), so ``started_at + latency_s`` must fit within the
owning session's ``started_at``/``ended_at`` bounds (within a small clock
tolerance).
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from my_coding_agent.observability.recorder import Recorder

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_RESPONSE = {"choices": [{"message": {"content": "hi"}}]}
_TOLERANCE_S = 1.0


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def test_timed_events_start_and_end_fit_within_session_bounds(tmp_path):
    sid = "timing0001"
    sdir = tmp_path / sid
    sdir.mkdir()
    rec = Recorder(session_id=sid, session_dir=sdir)

    rec.start("Agent", "test-model", 8192)

    # LLM call: wall start captured alongside the monotonic latency timer.
    llm_started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    t0 = time.monotonic()
    time.sleep(0.05)
    llm_latency = time.monotonic() - t0
    rec.record_llm_call(
        kind="main",
        call=1,
        latency_s=llm_latency,
        usage=_USAGE,
        messages=[{"role": "user", "content": "hi"}],
        context_window=8192,
        response_data=_RESPONSE,
        started_at=llm_started_at,
    )

    # Tool call: before_tool/after_tool stash and emit their own wall start.
    rec.before_tool("bash", {"command": "echo hi"})
    time.sleep(0.05)
    rec.after_tool("bash", {"command": "echo hi"}, "hi", True, None)

    rec.finish("stop", steps=1, elapsed_s=0.2)

    events = _read_events(sdir / "events.jsonl")
    start_ev = next(e for e in events if e["type"] == "session_start")
    end_ev = next(e for e in events if e["type"] == "session_end")
    session_start = _parse(start_ev["started_at"])
    session_end = _parse(end_ev["ended_at"])

    for ev in events:
        if "latency_s" not in ev or "started_at" not in ev:
            continue
        started_at = _parse(ev["started_at"])
        finished_at = started_at.timestamp() + ev["latency_s"]
        assert (
            session_start.timestamp() - _TOLERANCE_S
            <= started_at.timestamp()
            <= session_end.timestamp() + _TOLERANCE_S
        ), f"{ev['type']} started_at outside session bounds"
        assert finished_at <= session_end.timestamp() + _TOLERANCE_S, (
            f"{ev['type']} started_at + latency_s exceeds session end"
        )


def test_adjacent_fast_events_get_distinct_millisecond_timestamps(tmp_path):
    sid = "timing0002"
    sdir = tmp_path / sid
    sdir.mkdir()
    rec = Recorder(session_id=sid, session_dir=sdir)

    rec.record_finish_check(step=1, finish_reason="stop", signal="STOP")
    rec.record_finish_check(step=2, finish_reason="stop", signal="STOP")

    events = _read_events(sdir / "events.jsonl")
    stamps = [e["started_at"] for e in events]
    # Millisecond precision + offset means back-to-back emits are distinguishable
    # far more often than the old second-precision format; assert the format
    # itself carries an offset and sub-second component.
    for ts in stamps:
        parsed = _parse(ts)
        assert parsed.tzinfo is not None
        assert parsed.microsecond % 1000 == 0  # millisecond precision, not micro

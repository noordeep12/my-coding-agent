"""Skill observability: recorder event, reader analytics, envelope distinctions."""

import json
from pathlib import Path

from my_coding_agent.engine.tool_execution import ToolExecutor
from my_coding_agent.engine.tool_registry.skills import Skill
from my_coding_agent.observability import current_session_id
from my_coding_agent.observability.recorder import Recorder
from my_coding_agent.viewer.reader import load_session


def _ev(type: str, **kw):
    return {"type": type, **kw}


def _write_events(path: Path, events: list) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


# ── recorder ──────────────────────────────────────────────────────────────────


def test_record_skill_index_writes_event(tmp_path):
    rec = Recorder("sess0001", tmp_path)
    rec.record_skill_index(["a", "b", "c"], chars=345, tier="full")
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["type"] == "skill_index"
    assert ev["names"] == ["a", "b", "c"]
    assert ev["count"] == 3
    assert ev["chars"] == 345
    assert ev["tier"] == "full"


# ── reader analytics ──────────────────────────────────────────────────────────


def _skill_session(session_id="skillsess001") -> list:
    """A one-step session that offers a skill index and loads one skill."""
    return [
        _ev(
            "session_start",
            session_id=session_id,
            label="Test",
            model="gpt-4o-mini",
            context_window=8192,
            started_at="2026-01-01T10:00:00",
            parent_session_id=None,
        ),
        _ev(
            "skill_index",
            names=["a", "b", "c"],
            count=3,
            chars=200,
            tier="full",
            started_at="2026-01-01T10:00:00.5",
        ),
        _ev(
            "router",
            signal="s",
            selected=["use_skill"],
            phase="phase1_keyword",
            used_llm=False,
            started_at="2026-01-01T10:00:01",
        ),
        _ev(
            "llm_call",
            call=1,
            kind="main",
            latency_s=1.0,
            prompt=10,
            completion=5,
            total=15,
            context_window=8192,
            messages=None,
            response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="2026-01-01T10:00:02",
        ),
        _ev(
            "tool_call",
            name="use_skill",
            args={"name": "a"},
            result=json.dumps(
                {
                    "schema_version": 1,
                    "tool": "use_skill",
                    "ok": True,
                    "output": "Skill: a\n\nbody",
                    "error": None,
                    "metadata": {},
                }
            ),
            ok=True,
            latency_s=0.1,
            started_at="2026-01-01T10:00:03",
        ),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=1,
            elapsed_s=5.0,
            ended_at="2026-01-01T10:00:06",
        ),
    ]


def test_reader_surfaces_skill_offered_and_loaded_counts(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_events(path, _skill_session())
    session = load_session(path)
    assert session.analytics["skill_offered_count"] == 1
    assert session.analytics["skill_loaded_count"] == 1


def test_reader_skill_index_event_creates_no_node(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_events(path, _skill_session())
    session = load_session(path)
    assert not any(n.type == "skill_index" for n in session.nodes.values())


def test_skill_free_trace_has_no_skill_analytics(tmp_path):
    events = [
        _ev(
            "session_start",
            session_id="plain00000001",
            label="T",
            model="m",
            context_window=8192,
            started_at="t0",
            parent_session_id=None,
        ),
        _ev(
            "router",
            signal="s",
            selected=["bash"],
            phase="phase1_keyword",
            used_llm=False,
            started_at="t1",
        ),
        _ev(
            "llm_call",
            call=1,
            kind="main",
            latency_s=1.0,
            prompt=10,
            completion=5,
            total=15,
            context_window=8192,
            messages=None,
            response={"content": "done", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="t2",
        ),
        _ev("session_end", stop_reason="stop", steps=1, elapsed_s=5.0, ended_at="t3"),
    ]
    path = tmp_path / "events.jsonl"
    _write_events(path, events)
    session = load_session(path)
    assert "skill_offered_count" not in session.analytics
    assert "skill_loaded_count" not in session.analytics


# ── envelope distinguishability: load / dedup / error ─────────────────────────


def _use_skill_call(call_id: str, name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "use_skill", "arguments": json.dumps({"name": name})},
    }


def test_load_dedup_error_distinguishable_in_events(tmp_path, bare_llm):
    """Three use_skill calls in one message produce three distinguishable events."""
    rec = Recorder("sessSkill01", tmp_path)
    bare_llm._recorder = rec
    bare_llm._session_log_path = str(tmp_path / "log")
    message = {
        "tool_calls": [
            _use_skill_call("1", "a"),  # load
            _use_skill_call("2", "a"),  # dedup
            _use_skill_call("3", "missing"),  # error
        ]
    }
    token = current_session_id.set("sessSkill01")
    try:
        executor = ToolExecutor(
            message,
            bare_llm,
            tools=[],
            skills={"a": Skill("a", "does a", "FULL BODY A")},
            loaded_skills=set(),
        )
        executor.run()
    finally:
        current_session_id.reset(token)

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 3
    load_ev, dedup_ev, error_ev = tool_events

    # All read from the envelope only (ok verdict + output content), no free-text
    # parsing beyond the structured result (skill-event-capture).
    load_out = json.loads(load_ev["result"])
    assert load_ev["ok"] is True
    assert load_out["output"].startswith("Skill: a")
    assert "FULL BODY A" in load_out["output"]

    dedup_out = json.loads(dedup_ev["result"])
    assert dedup_ev["ok"] is True
    assert "already loaded" in dedup_out["output"].lower()
    assert "FULL BODY A" not in dedup_out["output"]

    assert error_ev["ok"] is False  # unknown skill → ok:false envelope

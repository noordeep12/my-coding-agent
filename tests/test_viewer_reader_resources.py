"""Tests that the reader surfaces `resources` on TraceNode and the run-level
rollup in analytics, and that resource-free legacy sessions still load
(node-resource-monitoring).
"""

from __future__ import annotations

import json
from pathlib import Path

from my_coding_agent.viewer.reader import load_session


def _ev(type_: str, **kw):
    return {"type": type_, **kw}


def _write_events(path: Path, events: list) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


_RESOURCES = {
    "machine_wide": True,
    "sample_count": 2,
    "ram_pct": {"avg": 10.0, "peak": 12.0},
    "cpu_pct": {"avg": 20.0, "peak": 25.0},
    "net_bytes": 100,
    "disk_bytes": 200,
}


def _events_with_resources(session_id: str = "resnode001") -> list:
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
            "router",
            signal="do stuff",
            selected=["bash"],
            phase="phase1_keyword",
            used_llm=False,
            started_at="2026-01-01T10:00:01",
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
            started_at="2026-01-01T10:00:02",
            resources=_RESOURCES,
        ),
        _ev(
            "tool_call",
            name="bash",
            args={"command": "echo hi"},
            result="hi",
            latency_s=0.1,
            started_at="2026-01-01T10:00:03",
            ok=True,
            resources=_RESOURCES,
        ),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=1,
            elapsed_s=2.0,
            ended_at="2026-01-01T10:00:04",
        ),
    ]


class TestNodeResources:
    def test_llm_call_node_carries_resources(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _events_with_resources())
        session = load_session(events_path)
        llm_nodes = [n for n in session.nodes.values() if n.type == "llm_call"]
        assert len(llm_nodes) == 1
        assert llm_nodes[0].attributes["resources"] == _RESOURCES

    def test_tool_call_node_carries_resources(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _events_with_resources())
        session = load_session(events_path)
        tool_nodes = [n for n in session.nodes.values() if n.type == "tool_call"]
        assert len(tool_nodes) == 1
        assert tool_nodes[0].attributes["resources"] == _RESOURCES


class TestLegacyEventsWithoutResources:
    def test_events_without_resources_load_unchanged(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        events = _events_with_resources()
        for ev in events:
            ev.pop("resources", None)
        _write_events(events_path, events)
        session = load_session(events_path)
        llm_nodes = [n for n in session.nodes.values() if n.type == "llm_call"]
        assert llm_nodes[0].attributes["resources"] is None
        assert "resource_rollup" not in session.analytics


class TestRunLevelRollup:
    def test_analytics_carries_resource_rollup_from_session_data(self, tmp_path):
        session_id = "resroll001"
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _events_with_resources(session_id))
        rollup = {
            "machine_wide": True,
            "sample_count": 10,
            "ram_pct": {"avg": 11.0, "peak": 15.0},
            "cpu_pct": {"avg": 22.0, "peak": 33.0},
            "net_bytes": 1000,
            "disk_bytes": 2000,
        }
        (tmp_path / "session_data.json").write_text(
            json.dumps({"resource_rollup": rollup})
        )
        session = load_session(events_path)
        assert session.analytics["resource_rollup"] == rollup

    def test_missing_session_data_json_omits_rollup(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, _events_with_resources())
        session = load_session(events_path)
        assert "resource_rollup" not in session.analytics

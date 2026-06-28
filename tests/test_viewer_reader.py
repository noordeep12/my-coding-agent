"""Tests for viewer/reader.py."""

import json
from pathlib import Path

from my_coding_agent.viewer.reader import (
    _detect_loops,
    _group_into_steps,
    _role_split,
    list_sessions,
    load_session,
)
from my_coding_agent.viewer.schema import TraceNode

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _ev(type: str, **kw):
    """Build a minimal event dict."""
    return {"type": type, **kw}


def _write_events(path: Path, events: list) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def _minimal_events(session_id: str = "aabbccdd1234") -> list:
    """Two-step session with router→llm_call→tool_call per step."""
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
        ),
        _ev(
            "tool_call",
            name="bash",
            args={"command": "echo hi"},
            result="hi",
            latency_s=0.1,
            started_at="2026-01-01T10:00:03",
        ),
        _ev(
            "token_tracking",
            step=1,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ctx_pct=1.2,
            context_window=8192,
            started_at="2026-01-01T10:00:03.5",
        ),
        _ev(
            "finish_check",
            step=1,
            finish_reason="tool_use",
            signal="CONTINUE",
            started_at="2026-01-01T10:00:03.6",
        ),
        _ev(
            "router",
            signal="next step",
            selected=["bash"],
            phase="phase1_keyword",
            used_llm=False,
            started_at="2026-01-01T10:00:04",
        ),
        _ev(
            "llm_call",
            call=2,
            kind="main",
            latency_s=0.8,
            prompt=200,
            completion=30,
            total=230,
            context_window=8192,
            messages=None,
            response={"content": "done", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="2026-01-01T10:00:05",
        ),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=2,
            elapsed_s=5.0,
            ended_at="2026-01-01T10:00:06",
        ),
    ]


def _composition_events(session_id: str = "aabbccdd1234") -> list:
    """Two-step session whose first LLM call carries a real message snapshot."""
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
            "router",
            signal="go",
            selected=["bash"],
            phase="p1",
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
            messages=[
                {"role": "system", "content": "s" * 80},
                {"role": "user", "content": "u" * 20},
            ],
            response={"content": "ok", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="2026-01-01T10:00:02",
        ),
        _ev(
            "tool_call",
            name="bash",
            args={"command": "echo hi"},
            result="hello world",
            latency_s=0.1,
            started_at="2026-01-01T10:00:03",
        ),
        _ev(
            "router",
            signal="next",
            selected=["bash"],
            phase="p1",
            used_llm=False,
            started_at="2026-01-01T10:00:04",
        ),
        _ev(
            "llm_call",
            call=2,
            kind="main",
            latency_s=0.8,
            prompt=200,
            completion=30,
            total=230,
            context_window=8192,
            messages=None,
            response={"content": "done", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="2026-01-01T10:00:05",
        ),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=2,
            elapsed_s=5.0,
            ended_at="2026-01-01T10:00:06",
        ),
    ]


# ── _group_into_steps ─────────────────────────────────────────────────────────


class TestGroupIntoSteps:
    def test_empty(self):
        assert _group_into_steps([]) == []

    def test_no_router_events(self):
        events = [_ev("session_start"), _ev("session_end")]
        assert _group_into_steps(events) == []

    def test_single_step(self):
        events = [
            _ev("session_start"),
            _ev("router", signal="x", selected=[], phase="p1", used_llm=False),
            _ev("llm_call", call=1, kind="main"),
        ]
        groups = _group_into_steps(events)
        assert len(groups) == 1
        assert groups[0][0]["type"] == "router"
        assert groups[0][1]["type"] == "llm_call"

    def test_two_steps(self):
        events = _minimal_events()
        groups = _group_into_steps(events)
        assert len(groups) == 2
        assert all(g[0]["type"] == "router" for g in groups)

    def test_events_before_first_router_are_dropped(self):
        events = [
            _ev("session_start"),
            _ev("handoff"),
            _ev("router", signal="s", selected=[], phase="p1", used_llm=False),
            _ev("llm_call", call=1, kind="main"),
        ]
        groups = _group_into_steps(events)
        assert len(groups) == 1
        assert groups[0][0]["type"] == "router"


# ── _detect_loops ─────────────────────────────────────────────────────────────


def _tool_node(id: str, name: str, args: dict) -> TraceNode:
    return TraceNode(
        id=id,
        type="tool_call",
        label=name,
        inputs={"args": args},
        outputs={},
        attributes={"name": name},
    )


class TestDetectLoops:
    def test_no_repeats_no_flag(self):
        nodes = {
            "a": _tool_node("a", "bash", {"command": "ls"}),
            "b": _tool_node("b", "bash", {"command": "pwd"}),
        }
        _detect_loops(nodes)
        assert not nodes["a"].loop_flag
        assert not nodes["b"].loop_flag

    def test_exact_repeat_flags_both(self):
        nodes = {
            "a": _tool_node("a", "bash", {"command": "ls"}),
            "b": _tool_node("b", "bash", {"command": "ls"}),
        }
        _detect_loops(nodes)
        assert nodes["a"].loop_flag
        assert nodes["b"].loop_flag

    def test_three_same_all_flagged(self):
        nodes = {
            "a": _tool_node("a", "read_file", {"path": "/foo"}),
            "b": _tool_node("b", "read_file", {"path": "/foo"}),
            "c": _tool_node("c", "read_file", {"path": "/foo"}),
        }
        _detect_loops(nodes)
        assert all(nodes[k].loop_flag for k in "abc")

    def test_different_tool_names_no_flag(self):
        nodes = {
            "a": _tool_node("a", "bash", {"command": "ls"}),
            "b": _tool_node("b", "read_file", {"path": "/foo"}),
        }
        _detect_loops(nodes)
        assert not nodes["a"].loop_flag
        assert not nodes["b"].loop_flag

    def test_non_tool_nodes_ignored(self):
        llm = TraceNode(
            id="x",
            type="llm_call",
            label="LLM",
            inputs={},
            outputs={},
            attributes={},
        )
        _detect_loops({"x": llm})
        assert not llm.loop_flag


# ── list_sessions ─────────────────────────────────────────────────────────────


class TestListSessions:
    def test_empty_dir(self, tmp_path):
        assert list_sessions(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_sessions(tmp_path / "missing") == []

    def test_dir_without_event_files(self, tmp_path):
        (tmp_path / "abc123").mkdir()
        assert list_sessions(tmp_path) == []

    def test_reads_events_jsonl(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        _write_events(sdir / "events.jsonl", _minimal_events(sid))
        rows = list_sessions(tmp_path)
        assert len(rows) == 1
        assert rows[0]["session_id"] == sid
        assert rows[0]["model"] == "gpt-4o-mini"

    def test_falls_back_to_session_data(self, tmp_path):
        sid = "deadbeef0000"
        sdir = tmp_path / sid
        sdir.mkdir()
        (sdir / "session_data.json").write_text(
            json.dumps(
                {
                    "session_id": sid,
                    "label": "X",
                    "model": "m",
                    "started_at": "2026-01-01T00:00:00",
                }
            ),
            encoding="utf-8",
        )
        rows = list_sessions(tmp_path)
        assert len(rows) == 1
        assert rows[0]["session_id"] == sid


# ── load_session ──────────────────────────────────────────────────────────────


class TestLoadSession:
    def test_happy_path(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))
        session = load_session(ep)
        assert session.session_id == sid
        assert session.steps == 2
        assert session.model == "gpt-4o-mini"
        assert len(session.nodes) > 0
        assert len(session.order) == len(session.nodes)

    def test_analytics_populated(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))
        session = load_session(ep)
        assert session.analytics["llm_call_count"] == 2
        assert session.analytics["tool_call_count"] == 1
        assert session.analytics["total_tokens"] == 380  # 150 + 230

    def test_fallback_when_no_events_jsonl(self, tmp_path):
        sid = "deadbeef0000"
        sdir = tmp_path / sid
        sdir.mkdir()
        (sdir / "session_data.json").write_text(
            json.dumps(
                {
                    "session_id": sid,
                    "label": "Y",
                    "model": "m",
                    "started_at": "2026-01-01",
                    "stop_reason": "stop",
                    "steps": 3,
                }
            ),
            encoding="utf-8",
        )
        session = load_session(sdir / "events.jsonl")
        assert session.session_id == sid
        assert session.analytics.get("source") == "session_data_fallback"
        assert len(session.nodes) == 2

    def test_loop_detection_flagged(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        events = [
            _ev(
                "session_start",
                session_id=sid,
                label="T",
                model="m",
                context_window=8192,
                started_at="2026-01-01T00:00:00",
                parent_session_id=None,
            ),
            _ev(
                "router",
                signal="s",
                selected=["bash"],
                phase="p1",
                used_llm=False,
                started_at="2026-01-01T00:00:01",
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
                started_at="2026-01-01T00:00:02",
            ),
            _ev(
                "tool_call",
                name="bash",
                args={"command": "ls"},
                result="",
                latency_s=0.1,
                started_at="2026-01-01T00:00:03",
            ),
            _ev(
                "router",
                signal="s2",
                selected=["bash"],
                phase="p1",
                used_llm=False,
                started_at="2026-01-01T00:00:04",
            ),
            _ev(
                "llm_call",
                call=2,
                kind="main",
                latency_s=1.0,
                prompt=10,
                completion=5,
                total=15,
                context_window=8192,
                messages=None,
                response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
                started_at="2026-01-01T00:00:05",
            ),
            _ev(
                "tool_call",
                name="bash",
                args={"command": "ls"},
                result="",
                latency_s=0.1,
                started_at="2026-01-01T00:00:06",
            ),
            _ev(
                "session_end",
                stop_reason="stop",
                steps=2,
                elapsed_s=2.0,
                ended_at="2026-01-01T00:00:07",
            ),
        ]
        _write_events(sdir / "events.jsonl", events)
        session = load_session(sdir / "events.jsonl")
        assert session.analytics["loop_count"] > 0
        looped = [n for n in session.nodes.values() if n.loop_flag]
        assert len(looped) == 2

    def test_no_step_wrapper_nodes(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))
        session = load_session(ep)
        assert all(n.type != "step" for n in session.nodes.values())
        # Step number survives as an attribute on the pipeline nodes.
        routers = [n for n in session.nodes.values() if n.type == "router"]
        assert {n.attributes["step"] for n in routers} == {1, 2}

    def test_order_is_execution_spine(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))
        session = load_session(ep)
        assert session.order[0] == f"{sid}::session"
        assert session.order[-1] == f"{sid}::session_end"
        assert len(session.order) == len(session.nodes)

    def test_ctx_state_tracks_composition_and_added(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _composition_events(sid))
        session = load_session(ep)

        # Session seeds the window with the first call's system + user input.
        sess = session.nodes[f"{sid}::session"]
        assert sess.ctx_state["composition"] == {"system": 80, "user": 20}
        assert sess.ctx_state["added"] == {"system": 80, "user": 20}

        # The LLM call appends its own assistant output (completion tokens).
        llm1 = session.nodes[f"{sid}::step1::llm::1"]
        assert llm1.ctx_state["added"] == {"assistant": 50}
        assert llm1.ctx_state["composition"]["assistant"] == 50
        assert llm1.ctx_state["window"] == 8192

        # A tool dispatch appends an estimated tool-result token figure.
        tool1 = session.nodes[f"{sid}::step1::tool::1"]
        assert tool1.ctx_state["added"].get("tool", 0) > 0
        assert tool1.ctx_state["estimated"] is True

        # A router adds nothing to the window.
        router2 = session.nodes[f"{sid}::step2::router"]
        assert router2.ctx_state["added_total"] == 0
        assert router2.ctx_state["removed"] == 0

    def test_circular_delegate_guard(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))
        # Pass sid in _seen to simulate a circular reference
        session = load_session(ep, _seen={sid})
        assert session.label == "[recursive]"


class TestRoleSplit:
    def test_splits_across_four_roles_and_sums_to_total(self):
        messages = [
            {"role": "system", "content": "x" * 100},
            {"role": "user", "content": "y" * 60},
            {"role": "assistant", "content": "z" * 40},
            {"role": "tool", "content": "t" * 200},
        ]
        split = _role_split(messages, 1000)
        # 400 chars total → system 25%, user 15%, assistant 10%, tool 50%.
        assert split == {"system": 250, "user": 150, "assistant": 100, "tool": 500}
        assert sum(split.values()) == 1000

    def test_drift_absorbed_so_parts_equal_total(self):
        # Char shares that don't divide evenly must still sum to the total.
        messages = [
            {"role": "system", "content": "x" * 33},
            {"role": "user", "content": "y" * 33},
            {"role": "tool", "content": "t" * 34},
        ]
        split = _role_split(messages, 1000)
        assert sum(split.values()) == 1000

    def test_unknown_role_counts_as_user(self):
        messages = [{"role": "developer", "content": "hi there"}]
        assert _role_split(messages, 50) == {"user": 50}

    def test_non_string_content_counted_via_json(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert _role_split(messages, 50) == {"user": 50}

    def test_returns_none_without_messages_or_tokens(self):
        assert _role_split([], 100) is None
        assert _role_split([{"role": "user", "content": "hi"}], None) is None
        assert _role_split([{"role": "user", "content": "hi"}], 0) is None

"""Tests for viewer/reader.py."""

import json
from pathlib import Path

from my_coding_agent.viewer.reader import (
    _detect_loops,
    _flag_anomalies,
    _group_into_steps,
    _read_events,
    _resolve_message_deltas,
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


def _delegate_parent_events(parent: str, child: str) -> list:
    """Main-agent session that delegates one step to *child*."""
    return [
        _ev(
            "session_start",
            session_id=parent,
            label="Main Agent",
            model="m",
            context_window=8192,
            started_at="2026-01-01T10:00:00",
            parent_session_id=None,
        ),
        _ev("router", signal="go", selected=["delegate"], phase="p1", used_llm=False),
        _ev(
            "llm_call",
            call=1,
            kind="main",
            prompt=100,
            completion=50,
            total=150,
            context_window=8192,
            messages=[
                {"role": "system", "content": "s" * 80},
                {"role": "user", "content": "u" * 20},
            ],
            response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
        ),
        _ev(
            "tool_call",
            name="delegate",
            args={"task": "explore"},
            result="report",
            child_session_id=child,
        ),
        _ev("session_end", stop_reason="stop", steps=1, elapsed_s=1.0),
    ]


def _subagent_events(child: str, parent: str) -> list:
    """Spawned sub-agent session with its own (smaller) context window."""
    return [
        _ev(
            "session_start",
            session_id=child,
            label="Subagent",
            model="m",
            context_window=8192,
            started_at="2026-01-01T10:00:02",
            parent_session_id=parent,
        ),
        _ev("router", signal="go", selected=["bash"], phase="p1", used_llm=False),
        _ev(
            "llm_call",
            call=1,
            kind="main",
            prompt=60,
            completion=30,
            total=90,
            context_window=8192,
            messages=[
                {"role": "system", "content": "s" * 40},
                {"role": "user", "content": "u" * 20},
            ],
            response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
        ),
        _ev("session_end", stop_reason="stop", steps=1, elapsed_s=1.0),
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


# ── Pre-change timestamp compatibility ──────────────────────────────────────────


class TestPreChangeTimestampCompatibility:
    """The reader must load traces recorded with the pre-change timestamp
    format (second precision, no timezone offset, emit-time stamp) without
    error — it treats ``started_at``/``ended_at`` as opaque strings, so no
    special-casing is required, but this is a regression guard on that
    invariant."""

    def test_old_format_naive_second_precision_trace_loads(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))  # uses naive, second-precision stamps
        session = load_session(ep)
        assert session.session_id == sid
        assert session.started_at == "2026-01-01T10:00:00"
        assert len(session.nodes) > 0

    def test_new_format_ms_offset_trace_loads(self, tmp_path):
        sid = "aabbccdd5678"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        events = _minimal_events(sid)
        for ev in events:
            for key in ("started_at", "ended_at"):
                if key in ev:
                    ev[key] = "2026-07-02T19:35:33.482+02:00"
        _write_events(ep, events)
        session = load_session(ep)
        assert session.session_id == sid
        assert len(session.nodes) > 0


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

    def test_finalize_step_merges_token_tracking_and_finish_check(self, tmp_path):
        # FinalizeStepNode emits token_tracking + finish_check as two events;
        # the trace must render them as a single finalize_step node with the
        # attributes of both combined.
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))
        session = load_session(ep)

        finalize = [n for n in session.nodes.values() if n.type == "finalize_step"]
        assert len(finalize) == 1
        assert not any(
            n.type in ("token_tracking", "finish_check") for n in session.nodes.values()
        )
        node = finalize[0]
        assert node.label == "FinalizeStepNode"
        # token-tracking attributes
        assert node.attributes["prompt_tokens"] == 100
        assert node.attributes["total_tokens"] == 150
        # finish-check attributes, on the same node
        assert node.attributes["finish_reason"] == "tool_use"
        assert node.attributes["signal"] == "CONTINUE"

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


class TestSubagentNesting:
    def _setup(self, tmp_path):
        parent, child = "parent000001", "child0000002"
        (tmp_path / parent).mkdir()
        (tmp_path / child).mkdir()
        _write_events(
            tmp_path / parent / "events.jsonl", _delegate_parent_events(parent, child)
        )
        _write_events(
            tmp_path / child / "events.jsonl", _subagent_events(child, parent)
        )
        return parent, child, load_session(tmp_path / parent / "events.jsonl")

    def test_nodes_carry_owning_agent_and_depth(self, tmp_path):
        parent, child, session = self._setup(tmp_path)
        # Main-agent nodes: agent == parent, depth 0 (root) / 1 (pipeline).
        assert session.nodes[f"{parent}::session"].agent == parent
        assert session.nodes[f"{parent}::session"].depth == 0
        assert session.nodes[f"{parent}::step1::llm::1"].depth == 1
        # Sub-agent nodes: agent == child, nested one level deeper.
        assert session.nodes[f"{child}::session"].agent == child
        assert session.nodes[f"{child}::session"].depth == 1
        assert session.nodes[f"{child}::step1::llm::1"].agent == child
        assert session.nodes[f"{child}::step1::llm::1"].depth == 2

    def test_subagent_root_nests_under_the_delegate_tool_call(self, tmp_path):
        """The child session's root must be re-parented to the exact `delegate`
        tool_call node — real parent/child linkage, not just depth/order."""
        parent, child, session = self._setup(tmp_path)
        delegate_node_id = f"{parent}::step1::tool::1"
        assert session.nodes[delegate_node_id].attributes["name"] == "delegate"
        assert session.nodes[f"{child}::session"].parent_id == delegate_node_id

    def test_context_windows_are_per_agent(self, tmp_path):
        parent, child, session = self._setup(tmp_path)
        # Parent window: system 80 + user 20 = 100, then +50 assistant = 150.
        parent_llm = session.nodes[f"{parent}::step1::llm::1"]
        assert parent_llm.ctx_state["tokens"] == 150
        # Child window is independent: 40 + 20 = 60, then +30 = 90 — NOT mixed in.
        child_llm = session.nodes[f"{child}::step1::llm::1"]
        assert child_llm.ctx_state["tokens"] == 90
        assert child_llm.ctx_state["composition"]["system"] == 40


class TestAnalyticsAllKindsAcrossTree:
    """Analytics totals cover every call kind across parent + embedded children
    (session-cost-attribution D5) — including a subagent's report call, which
    the pre-change reader counted nowhere."""

    def _setup(self, tmp_path):
        parent, child = "parent000001", "child0000002"
        (tmp_path / parent).mkdir()
        (tmp_path / child).mkdir()
        _write_events(
            tmp_path / parent / "events.jsonl", _delegate_parent_events(parent, child)
        )
        child_events = _subagent_events(child, parent)
        # Insert the child's end-of-turn report call before its session_end,
        # mirroring the evidence session's shape.
        child_events.insert(
            -1,
            _ev(
                "llm_call",
                call=2,
                kind="report",
                prompt=40,
                completion=10,
                total=50,
                context_window=8192,
                messages=None,
                response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
            ),
        )
        _write_events(tmp_path / child / "events.jsonl", child_events)
        return load_session(tmp_path / parent / "events.jsonl")

    def test_analytics_total_includes_report_tokens_from_embedded_child(self, tmp_path):
        session = self._setup(tmp_path)
        # parent main (150) + child main (90) + child report (50) = 290
        assert session.analytics["total_tokens"] == 290

    def test_by_kind_decomposes_across_the_tree(self, tmp_path):
        session = self._setup(tmp_path)
        by_kind = session.analytics["by_kind"]
        assert by_kind["main"]["total_tokens"] == 240  # 150 + 90
        assert by_kind["report"]["total_tokens"] == 50

    def test_by_agent_attributes_tokens_per_session(self, tmp_path):
        session = self._setup(tmp_path)
        by_agent = session.analytics["by_agent"]
        assert by_agent["parent000001"]["tokens"] == 150
        assert by_agent["child0000002"]["tokens"] == 140  # 90 + 50
        assert by_agent["child0000002"]["call_count"] == 2
        assert by_agent["child0000002"]["elapsed_s"] == 1.0


class TestAnalyticsBackwardCompat:
    """Pre-change traces (no rollup fields) and crash-truncated traces (no
    session_end / no session_data) still load, with analytics computed purely
    from event rows (D6)."""

    def test_pre_change_trace_loads_with_analytics(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, _minimal_events(sid))  # no by_kind/by_agent in the file
        session = load_session(ep)
        assert session.analytics["total_tokens"] == 380
        assert "by_kind" in session.analytics
        assert "by_agent" in session.analytics

    def test_crash_truncated_trace_no_session_end(self, tmp_path):
        sid = "crash000001"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        events = [e for e in _minimal_events(sid) if e.get("type") != "session_end"]
        _write_events(ep, events)
        session = load_session(ep)
        assert session.analytics["total_tokens"] == 380
        assert session.analytics["elapsed_s"] == 0.0

    def test_crash_truncated_trace_no_session_data_file(self, tmp_path):
        sid = "crash000002"
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        assert not ep.exists()
        assert not (sdir / "session_data.json").exists()
        session = load_session(ep)
        assert session.analytics == {"source": "session_data_fallback"}
        assert session.session_id == sid


class TestArtifactQueryNesting:
    """A read_tool_artifact call's internal artifact_query extraction call must
    nest under that exact tool_call node (recorder's `child_llm_calls` link),
    the same mechanism TestSubagentNesting exercises for delegate."""

    def _events(self, sid: str) -> list:
        return [
            _ev(
                "session_start",
                session_id=sid,
                label="Test",
                model="m",
                context_window=8192,
                started_at="2026-01-01T10:00:00",
                parent_session_id=None,
            ),
            _ev("router", signal="go", selected=["read_tool_artifact"], phase="p1"),
            _ev(
                "llm_call",
                call=1,
                kind="main",
                prompt=100,
                completion=50,
                total=150,
                context_window=8192,
                messages=[{"role": "user", "content": "u"}],
                response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
            ),
            # The extraction call happens inside read_tool_artifact's dispatch —
            # its llm_call event is emitted before the tool_call event, exactly
            # like the real ToolExecutor/registry flow.
            _ev(
                "llm_call",
                call=2,
                kind="artifact_query",
                prompt=20,
                completion=10,
                total=30,
                context_window=8192,
                messages=[{"role": "user", "content": "extract"}],
                response={
                    "content": "the detail",
                    "reasoning": "",
                    "tool_calls": [],
                    "raw": {},
                },
            ),
            _ev(
                "tool_call",
                name="read_tool_artifact",
                args={"tool_call_id": "call_1", "query": "the detail"},
                result="the detail",
                child_llm_calls=[2],
            ),
            _ev("session_end", stop_reason="stop", steps=1, elapsed_s=1.0),
        ]

    def test_artifact_query_llm_call_nests_under_its_tool_call(self, tmp_path):
        sid = "aabbccdd1234"
        sdir = tmp_path / sid
        sdir.mkdir()
        _write_events(sdir / "events.jsonl", self._events(sid))
        session = load_session(sdir / "events.jsonl")

        tool_node_id = f"{sid}::step1::tool::1"
        extraction_node_id = f"{sid}::step1::llm::2"
        tool_node = session.nodes[tool_node_id]
        extraction_node = session.nodes[extraction_node_id]

        assert tool_node.attributes["name"] == "read_tool_artifact"
        assert extraction_node.attributes["kind"] == "artifact_query"
        # Real parent/child link — not just sequential order.
        assert extraction_node.parent_id == tool_node_id
        assert extraction_node.depth == tool_node.depth + 1
        # The main llm_call stays flat under the session root, unaffected.
        assert session.nodes[f"{sid}::step1::llm::1"].parent_id == f"{sid}::session"


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


# ── Subagent report node ──────────────────────────────────────────────────────


class TestReportNode:
    """The ``report`` event becomes a distinct node, separate from ``handoff``."""

    def _load_with(self, tmp_path, sid, extra):
        # Report is emitted after the run finishes, so it follows session_end in
        # the stream — mirror that ordering here.
        events = _minimal_events(sid) + extra
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, events)
        return load_session(ep)

    def test_report_event_becomes_distinct_node(self, tmp_path):
        session = self._load_with(
            tmp_path,
            "reportsession",
            [
                _ev(
                    "report",
                    content="the final report",
                    source="verbatim",
                    started_at="2026-01-01T10:00:10",
                )
            ],
        )
        reports = [n for n in session.nodes.values() if n.type == "report"]
        assert len(reports) == 1
        node = reports[0]
        assert node.type == "report" and node.type != "handoff"
        assert node.label == "Subagent Report"
        assert node.outputs["content"] == "the final report"
        assert node.attributes["source"] == "verbatim"

    def test_no_report_node_without_report_event(self, tmp_path):
        session = self._load_with(tmp_path, "noreport", [])
        assert not any(n.type == "report" for n in session.nodes.values())

    def test_pre_provenance_report_event_surfaces_unknown_source(self, tmp_path):
        """A report event recorded before provenance existed (no ``source``
        key) still loads and is presented as unknown, never an error (D3)."""
        session = self._load_with(
            tmp_path,
            "prereport",
            [
                _ev(
                    "report",
                    content="an old report",
                    started_at="2026-01-01T10:00:10",
                )
            ],
        )
        reports = [n for n in session.nodes.values() if n.type == "report"]
        assert len(reports) == 1
        assert reports[0].attributes["source"] == "unknown"


class TestLlmToolDefinitions:
    """LLM-call nodes surface the tool definitions from the event's ``tools``."""

    def _load(self, tmp_path, sid, events):
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, events)
        return load_session(ep)

    def _llm_nodes(self, session):
        return [n for n in session.nodes.values() if n.type == "llm_call"]

    def test_tools_surfaced_in_inputs(self, tmp_path):
        sid = "tooldefs00001"
        tools = [{"type": "function", "function": {"name": "bash"}}]
        events = [
            _ev(
                "session_start",
                session_id=sid,
                label="T",
                model="gpt-4o-mini",
                context_window=8192,
                started_at="2026-01-01T10:00:00",
                parent_session_id=None,
            ),
            _ev(
                "router",
                signal="go",
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
                prompt=10,
                completion=5,
                total=15,
                context_window=8192,
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                response={
                    "content": "ok",
                    "reasoning": "",
                    "tool_calls": [],
                    "raw": {},
                },
                started_at="2026-01-01T10:00:02",
            ),
        ]
        session = self._load(tmp_path, sid, events)
        nodes = self._llm_nodes(session)
        assert len(nodes) == 1
        assert nodes[0].inputs["tools"] == tools

    def test_missing_tools_field_renders_empty(self, tmp_path):
        # _minimal_events predates tool capture — no ``tools`` key on the events.
        session = self._load(tmp_path, "aabbccdd1234", _minimal_events())
        nodes = self._llm_nodes(session)
        assert nodes
        assert all(n.inputs["tools"] == [] for n in nodes)

    def test_delta_encoded_call_shows_full_reconstructed_messages(self, tmp_path):
        sid = "tooldefs00002"
        m1 = [{"role": "user", "content": "hi"}]
        events = [
            _ev(
                "session_start",
                session_id=sid,
                label="T",
                model="gpt-4o-mini",
                context_window=8192,
                started_at="2026-01-01T10:00:00",
                parent_session_id=None,
            ),
            _ev(
                "router",
                signal="go",
                selected=["bash"],
                phase="phase1_keyword",
                used_llm=False,
                started_at="2026-01-01T10:00:01",
            ),
            _delta_llm_event(call=1, messages=m1),
            _delta_llm_event(
                call=2,
                messages=None,
                messages_base_call=1,
                messages_prefix_len=1,
                messages_suffix=[{"role": "assistant", "content": "a1"}],
            ),
        ]
        session = self._load(tmp_path, sid, events)
        nodes = self._llm_nodes(session)
        assert nodes[-1].inputs["messages"] == m1 + [
            {"role": "assistant", "content": "a1"}
        ]


def _delta_llm_event(**kw):
    base = {
        "type": "llm_call",
        "kind": "main",
        "latency_s": 1.0,
        "prompt": 10,
        "completion": 5,
        "total": 15,
        "context_window": 8192,
        "tools": [],
        "response": {"content": "ok", "reasoning": "", "tool_calls": [], "raw": {}},
        "started_at": "2026-01-01T10:00:02",
    }
    base.update(kw)
    return base


class TestResolveMessageDeltas:
    """Reader-side reconstruction of prefix-delta ``llm_call`` events (D3)."""

    def test_delta_reconstructs_full_snapshot(self):
        m1 = [{"role": "user", "content": "hi"}]
        events = [
            _delta_llm_event(call=1, messages=m1),
            _delta_llm_event(
                call=2,
                messages=None,
                messages_base_call=1,
                messages_prefix_len=1,
                messages_suffix=[{"role": "assistant", "content": "a1"}],
            ),
        ]
        _resolve_message_deltas(events)
        assert events[1]["messages"] == m1 + [{"role": "assistant", "content": "a1"}]

    def test_chained_deltas_resolve(self):
        m1 = [{"role": "user", "content": "hi"}]
        events = [
            _delta_llm_event(call=1, messages=m1),
            _delta_llm_event(
                call=2,
                messages=None,
                messages_base_call=1,
                messages_prefix_len=1,
                messages_suffix=[{"role": "assistant", "content": "a1"}],
            ),
            _delta_llm_event(
                call=3,
                messages=None,
                messages_base_call=2,
                messages_prefix_len=2,
                messages_suffix=[{"role": "user", "content": "u2"}],
            ),
        ]
        _resolve_message_deltas(events)
        assert events[2]["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]

    def test_legacy_full_snapshot_traces_are_unaffected(self):
        m1 = [{"role": "user", "content": "hi"}]
        m2 = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}]
        events = [
            _delta_llm_event(call=1, messages=m1),
            _delta_llm_event(call=2, messages=m2),
        ]
        _resolve_message_deltas(events)
        assert events[0]["messages"] == m1
        assert events[1]["messages"] == m2

    def test_missing_base_degrades_gracefully(self):
        events = [
            _delta_llm_event(
                call=2,
                messages=None,
                messages_base_call=1,
                messages_prefix_len=1,
                messages_suffix=[{"role": "assistant", "content": "a1"}],
            ),
        ]
        _resolve_message_deltas(events)
        assert events[0]["messages"] is None

    def test_truncated_file_reconstructs_completed_events(self, tmp_path):
        m1 = [{"role": "user", "content": "hi"}]
        events = [
            _delta_llm_event(call=1, messages=m1),
            _delta_llm_event(
                call=2,
                messages=None,
                messages_base_call=1,
                messages_prefix_len=1,
                messages_suffix=[{"role": "assistant", "content": "a1"}],
            ),
        ]
        path = tmp_path / "events.jsonl"
        lines = [json.dumps(e) for e in events]
        # Truncate mid-write of a third (never-completed) event.
        truncated = '{"type": "llm_call", "call"'
        path.write_text("\n".join(lines) + "\n" + truncated, encoding="utf-8")
        parsed = _read_events(path)
        assert len(parsed) == 2
        assert parsed[1]["messages"] == m1 + [{"role": "assistant", "content": "a1"}]


class TestCappedLlmCallBadge:
    """llm_call nodes derive a ``capped`` attribute from the recorded cap
    (extract-completeness-disclosure D6)."""

    def _load(self, tmp_path, sid, events):
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, events)
        return load_session(ep)

    def _llm_nodes(self, session):
        return [n for n in session.nodes.values() if n.type == "llm_call"]

    def _events(self, sid, llm_event):
        return [
            _ev(
                "session_start",
                session_id=sid,
                label="T",
                model="gpt-4o-mini",
                context_window=8192,
                started_at="2026-01-01T10:00:00",
                parent_session_id=None,
            ),
            _ev(
                "router",
                signal="go",
                selected=["bash"],
                phase="phase1_keyword",
                used_llm=False,
                started_at="2026-01-01T10:00:00.5",
            ),
            llm_event,
        ]

    def test_reader_marks_capped_llm_call(self, tmp_path):
        sid = "capped000001"
        events = self._events(
            sid,
            _ev(
                "llm_call",
                call=1,
                kind="artifact_query",
                latency_s=1.0,
                prompt=10,
                completion=800,
                total=810,
                context_window=8192,
                messages=None,
                max_tokens=800,
                response={
                    "content": "cut",
                    "reasoning": "",
                    "tool_calls": [],
                    "raw": {},
                    "finish_reason": None,
                },
                started_at="2026-01-01T10:00:01",
            ),
        )
        nodes = self._llm_nodes(self._load(tmp_path, sid, events))
        assert nodes[0].attributes["capped"] is True
        assert nodes[0].attributes["max_tokens"] == 800

    def test_reader_clean_call_not_capped(self, tmp_path):
        sid = "capped000002"
        events = self._events(
            sid,
            _ev(
                "llm_call",
                call=1,
                kind="artifact_query",
                latency_s=1.0,
                prompt=10,
                completion=10,
                total=20,
                context_window=8192,
                messages=None,
                max_tokens=800,
                response={
                    "content": "ok",
                    "reasoning": "",
                    "tool_calls": [],
                    "raw": {},
                    "finish_reason": "stop",
                },
                started_at="2026-01-01T10:00:01",
            ),
        )
        nodes = self._llm_nodes(self._load(tmp_path, sid, events))
        assert nodes[0].attributes["capped"] is False

    def test_reader_tolerates_missing_cap_field(self, tmp_path):
        # Old trace: no max_tokens field at all.
        session = self._load(tmp_path, "aabbccdd1234", _minimal_events())
        nodes = self._llm_nodes(session)
        assert nodes
        assert all(n.attributes["capped"] is False for n in nodes)


# ── Summarizer nesting ────────────────────────────────────────────────────────


def _summarizer_session_events(sid: str, kind: str, triggered_by: str) -> list:
    """One-step session whose step ends in a summarizer invocation."""
    events = [
        _ev(
            "session_start",
            session_id=sid,
            label="Test",
            model="m",
            context_window=8192,
            started_at="2026-01-01T10:00:00",
            parent_session_id=None,
        ),
        _ev(
            "router",
            signal="s",
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
            "token_tracking",
            step=1,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ctx_pct=1.2,
            context_window=8192,
            started_at="2026-01-01T10:00:03",
        ),
        _ev(
            "finish_check",
            step=1,
            finish_reason="tool_calls",
            signal="STOP",
            started_at="2026-01-01T10:00:03.5",
        ),
        _ev(
            "llm_call",
            call=2,
            kind=kind,
            latency_s=50.0,
            prompt=500,
            completion=80,
            total=580,
            context_window=8192,
            messages=None,
            response={
                "content": "summary",
                "reasoning": "",
                "tool_calls": [],
                "raw": {},
            },
            started_at="2026-01-01T10:00:04",
        ),
        _ev(
            "summarizer",
            kind=kind,
            step=1,
            triggered_by=triggered_by,
            latency_s=50.0,
            prompt_tokens=500,
            completion_tokens=80,
            total_tokens=580,
            started_at="2026-01-01T10:00:54",
        ),
    ]
    if triggered_by == "context_guard":
        events.append(
            _ev(
                "handoff",
                step=1,
                ctx_tokens=6000,
                ctx_pct=75.0,
                content="handoff text",
                path="/tmp/h.md",
            )
        )
    events.append(
        _ev(
            "session_end",
            stop_reason="max_steps",
            steps=1,
            elapsed_s=60.0,
            ended_at="2026-01-01T10:01:00",
        )
    )
    return events


class TestSummarizerNesting:
    def _load(self, tmp_path, sid, events):
        sdir = tmp_path / sid
        sdir.mkdir()
        _write_events(sdir / "events.jsonl", events)
        return load_session(sdir / "events.jsonl")

    def test_cutoff_summarizer_nests_under_finalize_step(self, tmp_path):
        sid = "aabbccdd1234"
        events = _summarizer_session_events(sid, "report", "finalize_step")
        session = self._load(tmp_path, sid, events)
        summ = session.nodes[f"{sid}::step1::summarizer"]
        finalize = session.nodes[f"{sid}::step1::finalize_step"]
        assert summ.label == "ContextSummarizerNode"
        assert summ.parent_id == finalize.id
        assert summ.depth == finalize.depth + 1
        # The report-kind LLM call nests beneath the summarizer.
        llm = session.nodes[f"{sid}::step1::llm::2"]
        assert llm.parent_id == summ.id
        assert llm.depth == summ.depth + 1
        # Execution order places the nested pair right after the trigger.
        idx = session.order.index(finalize.id)
        assert session.order[idx + 1] == summ.id
        assert session.order[idx + 2] == llm.id

    def test_handoff_summarizer_nests_under_context_guard(self, tmp_path):
        sid = "aabbccdd1234"
        events = _summarizer_session_events(sid, "handoff", "context_guard")
        session = self._load(tmp_path, sid, events)
        summ = session.nodes[f"{sid}::step1::summarizer"]
        guard = session.nodes[f"{sid}::step1::handoff"]
        assert guard.label == "ContextGuardNode"
        assert summ.parent_id == guard.id
        assert summ.depth == guard.depth + 1
        llm = session.nodes[f"{sid}::step1::llm::2"]
        assert llm.parent_id == summ.id

    def test_summarizer_without_trigger_node_stays_flat(self, tmp_path):
        # A context-limit stop records no handoff event, so the guard has no
        # node in the tree; the summarizer must fall back to the flat chain.
        sid = "aabbccdd1234"
        events = [
            e
            for e in _summarizer_session_events(sid, "report", "context_guard")
            if e["type"] != "handoff"
        ]
        session = self._load(tmp_path, sid, events)
        summ = session.nodes[f"{sid}::step1::summarizer"]
        root = session.order[0]
        assert summ.parent_id == root

    def test_legacy_trace_without_summarizer_renders_flat(self, tmp_path):
        # Pre-summarizer traces: a report-kind llm_call with no summarizer
        # event stays a flat child of the session root, exactly as before.
        sid = "aabbccdd1234"
        events = [
            e
            for e in _summarizer_session_events(sid, "report", "finalize_step")
            if e["type"] != "summarizer"
        ]
        session = self._load(tmp_path, sid, events)
        assert f"{sid}::step1::summarizer" not in session.nodes
        llm = session.nodes[f"{sid}::step1::llm::2"]
        assert llm.parent_id == session.order[0]


# ── Anomaly rendering ──────────────────────────────────────────────────────────


def _failed_tool_call(name: str, command: str, started_at: str) -> dict:
    result = json.dumps(
        {"schema_version": 1, "tool": name, "ok": False, "output": "boom"}
    )
    return _ev(
        "tool_call",
        name=name,
        args={"command": command},
        result=result,
        latency_s=0.1,
        started_at=started_at,
    )


def _anomaly_row(streak_id: str, streak_len: int, step: int, tokens_spent: int) -> dict:
    return _ev(
        "anomaly",
        kind="failure_streak",
        streak_id=streak_id,
        signature="bash|json.decoder.JSONDecodeError",
        tool_name="bash",
        streak_len=streak_len,
        tokens_spent=tokens_spent,
        step=step,
        started_at=f"2026-01-01T10:00:0{step}.9",
    )


def _streak_session_events(sid: str = "aabbccdd1234") -> list:
    """A 3-step session where every step's tool call fails, growing one streak."""
    events = [
        _ev(
            "session_start",
            session_id=sid,
            label="Test",
            model="gpt-4o-mini",
            context_window=8192,
            started_at="2026-01-01T10:00:00",
            parent_session_id=None,
        )
    ]
    for step in range(1, 4):
        events += [
            _ev(
                "router",
                signal="go",
                selected=["bash"],
                phase="p1",
                used_llm=False,
                started_at=f"2026-01-01T10:00:0{step}",
            ),
            _ev(
                "llm_call",
                call=step,
                kind="main",
                latency_s=1.0,
                prompt=10,
                completion=5,
                total=15,
                context_window=8192,
                messages=None,
                response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
                started_at=f"2026-01-01T10:00:0{step}.1",
            ),
            _failed_tool_call("bash", f"cmd{step}", f"2026-01-01T10:00:0{step}.2"),
            _anomaly_row("abc123-1", step, step, step * 15) if step >= 3 else None,
            _ev(
                "token_tracking",
                step=step,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                ctx_pct=1.0,
                context_window=8192,
                started_at=f"2026-01-01T10:00:0{step}.3",
            ),
            _ev(
                "finish_check",
                step=step,
                finish_reason="tool_use",
                signal="CONTINUE",
                started_at=f"2026-01-01T10:00:0{step}.4",
            ),
        ]
    events = [e for e in events if e is not None]
    events.append(
        _ev(
            "session_end",
            stop_reason="max_steps",
            steps=3,
            elapsed_s=1.0,
            ended_at="2026-01-01T10:00:10",
        )
    )
    return events


class TestAnomalyRendering:
    def _load(self, tmp_path, sid, events):
        sdir = tmp_path / sid
        sdir.mkdir()
        ep = sdir / "events.jsonl"
        _write_events(ep, events)
        return load_session(ep)

    def test_anomaly_node_created_with_final_magnitude(self, tmp_path):
        sid = "aabbccdd1234"
        session = self._load(tmp_path, sid, _streak_session_events(sid))
        node = session.nodes[f"{sid}::anomaly::abc123-1"]
        assert node.type == "anomaly"
        assert node.attributes["streak_len"] == 3
        assert node.attributes["tokens_spent"] == 45
        assert node.attributes["signature"] == "bash|json.decoder.JSONDecodeError"

    def test_only_latest_row_kept_no_duplicate_nodes(self, tmp_path):
        sid = "aabbccdd1234"
        session = self._load(tmp_path, sid, _streak_session_events(sid))
        anomaly_nodes = [n for n in session.nodes.values() if n.type == "anomaly"]
        assert len(anomaly_nodes) == 1

    def test_flags_exactly_the_streaks_tool_call_nodes(self, tmp_path):
        sid = "aabbccdd1234"
        session = self._load(tmp_path, sid, _streak_session_events(sid))
        tool_nodes = [n for n in session.nodes.values() if n.type == "tool_call"]
        assert len(tool_nodes) == 3
        assert all(n.anomaly_flag for n in tool_nodes)

    def test_analytics_anomaly_count(self, tmp_path):
        sid = "aabbccdd1234"
        session = self._load(tmp_path, sid, _streak_session_events(sid))
        assert session.analytics["anomaly_count"] == 1

    def test_session_without_anomaly_rows_unaffected(self, tmp_path):
        sid = "aabbccdd1234"
        session = self._load(tmp_path, sid, _minimal_events(sid))
        assert not any(n.type == "anomaly" for n in session.nodes.values())
        assert not any(n.anomaly_flag for n in session.nodes.values())
        assert session.analytics["anomaly_count"] == 0

    def test_tool_call_node_carries_top_level_ok_and_error_class(self, tmp_path):
        sid = "aabbccdd1234"
        session = self._load(tmp_path, sid, _streak_session_events(sid))
        tool_nodes = [n for n in session.nodes.values() if n.type == "tool_call"]
        # Pre-change fixture events carry no top-level ok — node attribute
        # reflects that absence rather than inventing a verdict.
        assert all(n.attributes["ok"] is None for n in tool_nodes)


class TestOutcomeFieldsFailureDetection:
    def test_new_style_trace_detects_failure_from_top_level_fields(self):
        node = _tool_node("a", "bash", {"command": "x"})
        node.outputs = {"result": "not json — must not be parsed"}
        node.attributes["ok"] = False
        nodes = {"a": node}
        events = [_anomaly_row("s-1", 1, 1, 10)]
        _flag_anomalies(nodes, ["a"], events)
        assert nodes["a"].anomaly_flag

    def test_pre_change_trace_falls_back_to_result_parse(self):
        node = _tool_node("a", "bash", {"command": "x"})
        node.outputs = {"result": json.dumps({"ok": False})}
        # No "ok" key set on attributes — mirrors a pre-change trace's node.
        nodes = {"a": node}
        events = [_anomaly_row("s-1", 1, 1, 10)]
        _flag_anomalies(nodes, ["a"], events)
        assert nodes["a"].anomaly_flag


class TestFlagAnomaliesUnit:
    def test_stops_at_streak_len_even_with_more_matching_failures_earlier(self):
        # Two failing bash nodes precede the streak's own 3, but the streak
        # only claims the trailing 3 (streak_len=3), matching detection's
        # consecutive-only semantics.
        nodes = {
            "a": _tool_node("a", "bash", {"command": "old1"}),
            "b": _tool_node("b", "bash", {"command": "old2"}),
            "c": _tool_node("c", "bash", {"command": "c1"}),
            "d": _tool_node("d", "bash", {"command": "c2"}),
            "e": _tool_node("e", "bash", {"command": "c3"}),
        }
        for n in nodes.values():
            n.outputs = {"result": json.dumps({"ok": False})}
        order = ["a", "b", "c", "d", "e"]
        events = [_anomaly_row("s-1", 3, 1, 30)]
        _flag_anomalies(nodes, order, events)
        assert not nodes["a"].anomaly_flag
        assert not nodes["b"].anomaly_flag
        assert nodes["c"].anomaly_flag
        assert nodes["d"].anomaly_flag
        assert nodes["e"].anomaly_flag

    def test_breaks_on_mismatching_tool_call(self):
        nodes = {
            "a": _tool_node("a", "bash", {"command": "c1"}),
            "b": _tool_node("b", "read_file", {"path": "/x"}),
        }
        nodes["a"].outputs = {"result": json.dumps({"ok": False})}
        nodes["b"].outputs = {"result": json.dumps({"ok": False})}
        order = ["a", "b"]
        events = [_anomaly_row("s-1", 2, 1, 30)]
        _flag_anomalies(nodes, order, events)
        assert not nodes["a"].anomaly_flag
        assert not nodes["b"].anomaly_flag

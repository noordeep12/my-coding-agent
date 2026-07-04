"""Regression tests for Recorder.record_token_tracking and record_finish_check."""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import (
    ANOMALY,
    FINISH_CHECK,
    HANDOFF,
    LLM_CALL,
    REPORT,
    TOKEN_TRACKING,
    Recorder,
)

_TOOLS = [{"type": "function", "function": {"name": "read_file"}}]
_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_RESPONSE = {"choices": [{"message": {"content": "hi"}}]}


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


class TestRecordReport:
    def test_emits_report_type(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_report(content="final report")
        ev = _read_events(path)[-1]
        assert ev["type"] == REPORT

    def test_report_type_is_distinct_from_handoff(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_handoff(
            step=1, ctx_tokens=100, ctx_pct=50.0, content="h", path="/tmp/h.md"
        )
        rec.record_report(content="r")
        events = _read_events(path)
        assert events[-2]["type"] == HANDOFF
        assert events[-1]["type"] == REPORT
        assert HANDOFF != REPORT

    def test_content_and_started_at_present(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_report(content="the report body")
        ev = _read_events(path)[-1]
        assert ev["content"] == "the report body"
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


class TestRecordLlmCallTools:
    def test_tools_kept_for_full_payload_kind(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_llm_call(
            kind="main",
            call=1,
            latency_s=0.1,
            usage=_USAGE,
            messages=[{"role": "user", "content": "hi"}],
            context_window=8192,
            response_data=_RESPONSE,
            tools=_TOOLS,
        )
        ev = _read_events(path)[-1]
        assert ev["type"] == LLM_CALL
        assert ev["tools"] == _TOOLS

    def test_tools_dropped_for_non_payload_kind(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_llm_call(
            kind="other",
            call=1,
            latency_s=0.1,
            usage=_USAGE,
            messages=[{"role": "user", "content": "hi"}],
            context_window=8192,
            response_data=_RESPONSE,
            tools=_TOOLS,
        )
        ev = _read_events(path)[-1]
        assert ev["tools"] is None
        assert ev["messages"] is None

    def test_tools_default_empty_when_omitted(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_llm_call(
            kind="main",
            call=1,
            latency_s=0.1,
            usage=_USAGE,
            messages=[],
            context_window=8192,
            response_data=_RESPONSE,
        )
        ev = _read_events(path)[-1]
        assert ev["tools"] == []


class TestChildLlmCallLinking:
    """An LLM call made *during* a tool's dispatch (e.g. read_tool_artifact's
    artifact_query extraction) is stashed and attached to that tool's own
    tool_call event, so the viewer can nest it under the exact tool node —
    the same 'stash now, attach at after_tool' pattern as the delegate
    child-session link (note_delegate_child)."""

    def _record_llm_call(self, rec, call):
        rec.record_llm_call(
            kind="artifact_query",
            call=call,
            latency_s=0.05,
            usage=_USAGE,
            messages=[{"role": "user", "content": "extract"}],
            context_window=8192,
            response_data=_RESPONSE,
        )

    def test_llm_call_during_tool_dispatch_is_attached_to_its_tool_event(
        self, tmp_path
    ):
        rec, path = _make_recorder(tmp_path)
        rec.before_tool("read_tool_artifact", {"tool_call_id": "c1", "query": "q"})
        self._record_llm_call(rec, call=1)
        rec.after_tool(
            "read_tool_artifact", {"tool_call_id": "c1", "query": "q"}, "result"
        )
        events = _read_events(path)
        tool_ev = next(e for e in events if e["type"] == "tool_call")
        assert tool_ev["child_llm_calls"] == [1]

    def test_llm_call_outside_tool_dispatch_is_not_attached(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        self._record_llm_call(rec, call=1)  # no tool pending
        rec.before_tool("bash", {"command": "ls"})
        rec.after_tool("bash", {"command": "ls"}, "result")
        events = _read_events(path)
        tool_ev = next(e for e in events if e["type"] == "tool_call")
        assert "child_llm_calls" not in tool_ev

    def test_pending_child_calls_reset_between_tool_calls(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.before_tool("read_tool_artifact", {})
        self._record_llm_call(rec, call=1)
        rec.after_tool("read_tool_artifact", {}, "result 1")
        rec.before_tool("bash", {"command": "ls"})  # no LLM call inside this one
        rec.after_tool("bash", {"command": "ls"}, "result 2")
        events = [e for e in _read_events(path) if e["type"] == "tool_call"]
        assert events[0]["child_llm_calls"] == [1]
        assert "child_llm_calls" not in events[1]


class TestRecordLlmCallMessageDeltas:
    """Prefix-delta emission (design D2): consecutive calls of the same kind
    that only append messages should emit a delta, not a full snapshot;
    anything that breaks the identity-verified prefix must fall back to a
    full snapshot."""

    def _call(self, rec, *, kind, call, messages):
        rec.record_llm_call(
            kind=kind,
            call=call,
            latency_s=0.1,
            usage=_USAGE,
            messages=messages,
            context_window=8192,
            response_data=_RESPONSE,
        )

    def test_first_call_of_a_kind_is_full_snapshot(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        msgs = [{"role": "user", "content": "hi"}]
        self._call(rec, kind="main", call=1, messages=msgs)
        ev = _read_events(path)[-1]
        assert ev["messages"] == msgs
        assert "messages_base_call" not in ev

    def test_appended_messages_emit_a_delta(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        m1 = [{"role": "user", "content": "hi"}]
        self._call(rec, kind="main", call=1, messages=m1)
        m2 = m1 + [{"role": "assistant", "content": "hello"}]
        self._call(rec, kind="main", call=2, messages=m2)
        ev = _read_events(path)[-1]
        assert ev["messages"] is None
        assert ev["messages_base_call"] == 1
        assert ev["messages_prefix_len"] == 1
        assert ev["messages_suffix"] == [{"role": "assistant", "content": "hello"}]

    def test_deltas_chain_across_three_calls(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        m1 = [{"role": "user", "content": "hi"}]
        self._call(rec, kind="main", call=1, messages=m1)
        m2 = m1 + [{"role": "assistant", "content": "a1"}]
        self._call(rec, kind="main", call=2, messages=m2)
        m3 = m2 + [{"role": "user", "content": "u2"}]
        self._call(rec, kind="main", call=3, messages=m3)
        ev = _read_events(path)[-1]
        assert ev["messages_base_call"] == 2
        assert ev["messages_prefix_len"] == 2
        assert ev["messages_suffix"] == [{"role": "user", "content": "u2"}]

    def test_replaced_history_falls_back_to_full_snapshot(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        m1 = [{"role": "user", "content": "hi"}]
        self._call(rec, kind="main", call=1, messages=m1)
        # Handoff: brand-new list of brand-new dict objects — identity break.
        m2 = [{"role": "system", "content": "reset"}]
        self._call(rec, kind="main", call=2, messages=m2)
        ev = _read_events(path)[-1]
        assert ev["messages"] == m2
        assert "messages_base_call" not in ev

    def test_shorter_history_falls_back_to_full_snapshot(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        m1 = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}]
        self._call(rec, kind="main", call=1, messages=m1)
        m2 = m1[:1]
        self._call(rec, kind="main", call=2, messages=m2)
        ev = _read_events(path)[-1]
        assert ev["messages"] == m2
        assert "messages_base_call" not in ev

    def test_snapshot_is_not_corrupted_by_later_in_place_appends(self, tmp_path):
        """Regression: the pipeline appends new messages to the *same*
        ``ctx.messages`` list object after each call returns (never replacing
        it), so the recorder must not keep a live reference to the caller's
        list — only a snapshot of it as of call time. Otherwise the stored
        "base" snapshot's length silently grows between calls, corrupting
        ``messages_prefix_len`` and losing messages on reconstruction."""
        rec, path = _make_recorder(tmp_path)
        live_messages = [{"role": "user", "content": "hi"}]
        self._call(rec, kind="main", call=1, messages=live_messages)
        # Simulate the pipeline appending in place to the *same* list object
        # (as llm_call.py / tool_dispatch.py do) between calls 1 and 2.
        live_messages.append({"role": "assistant", "content": "a1"})
        live_messages.append({"role": "tool", "content": "t1"})
        self._call(rec, kind="main", call=2, messages=live_messages)
        events = _read_events(path)
        assert events[0]["messages"] == [{"role": "user", "content": "hi"}]
        ev2 = events[1]
        assert ev2["messages"] is None
        assert ev2["messages_base_call"] == 1
        assert ev2["messages_prefix_len"] == 1
        assert ev2["messages_suffix"] == [
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1"},
        ]

    def test_interleaved_kinds_each_chain_independently(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        main1 = [{"role": "user", "content": "hi"}]
        self._call(rec, kind="main", call=1, messages=main1)
        router1 = [{"role": "user", "content": "route"}]
        self._call(rec, kind="tool_router", call=2, messages=router1)
        main2 = main1 + [{"role": "assistant", "content": "a1"}]
        self._call(rec, kind="main", call=3, messages=main2)
        events = _read_events(path)
        router_ev = events[1]
        main_ev = events[2]
        assert "messages_base_call" not in router_ev  # first call of its kind
        assert main_ev["messages_base_call"] == 1  # chains off main's call 1


class TestRecordAnomaly:
    def test_emits_correct_type_and_row_shape(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_anomaly(
            kind="failure_streak",
            streak_id="abc123-1",
            signature="bash|json.decoder.JSONDecodeError",
            tool_name="bash",
            streak_len=3,
            tokens_spent=41230,
            step=7,
        )
        ev = _read_events(path)[-1]
        assert ev["type"] == ANOMALY
        assert ev["kind"] == "failure_streak"
        assert ev["streak_id"] == "abc123-1"
        assert ev["signature"] == "bash|json.decoder.JSONDecodeError"
        assert ev["tool_name"] == "bash"
        assert ev["streak_len"] == 3
        assert ev["tokens_spent"] == 41230
        assert ev["step"] == 7
        assert "started_at" in ev

    def test_rows_of_one_streak_share_streak_id(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_anomaly(
            kind="failure_streak",
            streak_id="abc123-1",
            signature="bash|json.decoder.JSONDecodeError",
            tool_name="bash",
            streak_len=3,
            tokens_spent=10000,
            step=7,
        )
        rec.record_anomaly(
            kind="failure_streak",
            streak_id="abc123-1",
            signature="bash|json.decoder.JSONDecodeError",
            tool_name="bash",
            streak_len=5,
            tokens_spent=25000,
            step=9,
        )
        events = [e for e in _read_events(path) if e["type"] == ANOMALY]
        assert len(events) == 2
        assert events[0]["streak_id"] == events[1]["streak_id"] == "abc123-1"
        assert events[-1]["streak_len"] == 5
        assert events[-1]["tokens_spent"] == 25000

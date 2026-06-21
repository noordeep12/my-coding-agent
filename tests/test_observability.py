"""Tests for the observability capture + reader layer.

These exercise the Recorder → events.jsonl → reader roundtrip and every derived
view (context series, message diff, bottlenecks, loop detection, code extraction,
and the parent/child session tree) against tmp_path so no real .my_coding_agent/
directory is touched. No live LLM is required.
"""

from my_coding_agent.observability import reader
from my_coding_agent.observability.recorder import Recorder


def _fake_response(content="", reasoning="", tool_calls=None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls or [],
                }
            }
        ]
    }


def _record_session(session_dir, *, parent=None):
    """Drive a Recorder through a representative run and return the session_id."""
    rec = Recorder("sess-main", session_dir, parent_session_id=parent)
    rec.start("Main Agent", "local-model", context_window=1000)
    rec.record_llm_call(
        kind="main",
        call=1,
        latency_s=2.0,
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        context_window=1000,
        response_data=_fake_response(content="let me run ls"),
    )
    # tool hook pair (before then after) → one tool_call event
    rec.before_tool("bash", {"command": "ls"})
    rec.after_tool("bash", {"command": "ls"}, "a.py\nb.py")
    rec.record_llm_call(
        kind="main",
        call=2,
        latency_s=3.0,
        usage={"prompt_tokens": 250, "completion_tokens": 20, "total_tokens": 270},
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "let me run ls"},
            {"role": "tool", "content": "a.py\nb.py"},
        ],
        context_window=1000,
        response_data=_fake_response(content="done"),
    )
    rec.finish("stop", steps=2, elapsed_s=5.0)
    return rec


# --- roundtrip ----------------------------------------------------------------


def test_recorder_writes_jsonl(tmp_path):
    _record_session(tmp_path)
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    # start + 2 llm + 1 tool + end
    assert len(lines) == 5


def test_load_session_parses_events(tmp_path):
    _record_session(tmp_path)
    session = reader.load_session(tmp_path)
    assert session is not None
    assert session.label == "Main Agent"
    assert session.stop_reason == "stop"
    assert session.ok is True
    assert len(session.llm_calls) == 2
    assert len(session.tool_calls) == 1
    assert session.total_tokens == 110 + 270


def test_load_session_returns_none_without_events(tmp_path):
    assert reader.load_session(tmp_path) is None


def test_load_session_tolerates_corrupt_final_line(tmp_path):
    _record_session(tmp_path)
    with (tmp_path / "events.jsonl").open("a") as fh:
        fh.write('{"type": "llm_call", "call": 3,')  # truncated/crashed write
    session = reader.load_session(tmp_path)
    assert session is not None
    assert len(session.llm_calls) == 2  # corrupt line skipped


# --- ancillary kinds omit the message snapshot --------------------------------


def test_ancillary_kind_keeps_messages(tmp_path):
    # Ancillary calls now keep their input so the viewer can show input/output.
    rec = Recorder("s", tmp_path)
    rec.record_llm_call(
        kind="tool_router",
        call=1,
        latency_s=0.1,
        usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        messages=[{"role": "user", "content": "x"}],
        context_window=1000,
        response_data=_fake_response(),
    )
    session = reader.load_session(tmp_path)
    assert session.llm_calls[0].messages == [{"role": "user", "content": "x"}]


# --- derived views ------------------------------------------------------------


def test_context_series(tmp_path):
    _record_session(tmp_path)
    series = reader.context_series(reader.load_session(tmp_path))
    assert series["prompt_tokens"] == [100, 250]
    assert series["pct"] == [10.0, 25.0]


def test_message_diff_detects_added_messages(tmp_path):
    _record_session(tmp_path)
    session = reader.load_session(tmp_path)
    diff = reader.message_diff(session.llm_calls[0], session.llm_calls[1])
    added = [line for tag, line in diff if tag == "+"]
    assert any("assistant" in line for line in added)


def test_bottlenecks_aggregates_and_costs_zero_for_local(tmp_path):
    _record_session(tmp_path)
    rows = reader.bottlenecks(reader.load_session(tmp_path))
    by_step = {r["step"]: r for r in rows}
    assert by_step["llm:main"]["calls"] == 2
    assert by_step["llm:main"]["tokens"] == 380
    assert by_step["bash"]["calls"] == 1
    assert all(r["cost_usd"] == 0.0 for r in rows)  # local model is free


def test_detect_loops_flags_repeated_tool(tmp_path):
    rec = Recorder("s", tmp_path)
    rec.start("A", "m", 1000)
    for _ in range(3):
        rec.before_tool("bash", {"command": "ls"})
        rec.after_tool("bash", {"command": "ls"}, "out")
    rec.finish("stop", 3, 1.0)
    flags = reader.detect_loops(reader.load_session(tmp_path))
    tool_flags = [f for f in flags if f["kind"] == "tool"]
    assert tool_flags and tool_flags[0]["count"] == 3


def test_extract_code_blocks(tmp_path):
    rec = Recorder("s", tmp_path)
    rec.before_tool("write_file", {"file_path": "x.py", "content": "print(1)"})
    rec.after_tool("write_file", {"file_path": "x.py", "content": "print(1)"}, "ok")
    rec.finish("stop", 1, 1.0)
    blocks = reader.extract_code_blocks(reader.load_session(tmp_path))
    assert blocks[0]["language"] == "python"
    assert "print(1)" in blocks[0]["code"]


# --- tree ---------------------------------------------------------------------


def test_load_all_sessions_nests_children(tmp_path):
    root = tmp_path / ".my_coding_agent"
    parent_dir = root / "parent"
    child_dir = root / "child"
    Recorder("parent", parent_dir).start("Main", "m", 1000)
    (parent_dir / "events.jsonl")  # ensure file exists via start above
    child_rec = Recorder("child", child_dir, parent_session_id="parent")
    child_rec.start("SubAgent", "m", 1000)
    sessions = reader.load_all_sessions(root)
    assert len(sessions) == 1
    assert sessions[0].session_id == "parent"
    assert [c.session_id for c in sessions[0].children] == ["child"]


def test_load_all_sessions_empty_root(tmp_path):
    assert reader.load_all_sessions(tmp_path / "missing") == []


def test_priced_model_computes_cost(tmp_path, monkeypatch):
    from my_coding_agent.observability import pricing

    monkeypatch.setitem(pricing.PRICES, "paid-model", (1.0, 2.0))
    rec = Recorder("s", tmp_path)
    rec.start("A", "paid-model", 1000)
    rec.record_llm_call(
        kind="main",
        call=1,
        latency_s=1.0,
        usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        messages=[{"role": "user", "content": "x"}],
        context_window=1000,
        response_data=_fake_response(),
    )
    rec.finish("stop", 1, 1.0)
    rows = reader.bottlenecks(reader.load_session(tmp_path))
    # 1000/1k*1.0 + 500/1k*2.0 = 2.0
    assert rows[0]["cost_usd"] == 2.0


def test_extract_fenced_block_from_assistant(tmp_path):
    rec = Recorder("s", tmp_path)
    rec.record_llm_call(
        kind="main",
        call=1,
        latency_s=1.0,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        messages=[{"role": "user", "content": "x"}],
        context_window=1000,
        response_data=_fake_response(content="here:\n```python\nx = 1\n```\n"),
    )
    rec.finish("stop", 1, 1.0)
    blocks = reader.extract_code_blocks(reader.load_session(tmp_path))
    assert any("x = 1" in b["code"] and b["language"] == "python" for b in blocks)


def test_tool_status_joined_from_session_data(tmp_path):
    _record_session(tmp_path)
    (tmp_path / "session_data.json").write_text(
        '{"tool_records": [{"ok": true, "status": "success", '
        '"tool_call_id": "call_1"}]}'
    )
    session = reader.load_session(tmp_path)
    assert session.tool_calls[0].ok is True
    assert session.tool_calls[0].tool_call_id == "call_1"

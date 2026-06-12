"""Tests for the terminal-UI renderers in logger.py.

These call the box-drawing functions and assert on the *structure and content*
of what they produce — that expected labels and values appear, that the row
helpers return the right number of rows, and that long inputs wrap/truncate
(CONTRIBUTE.md §42: assert behavior, do not merely execute for coverage). All
ANSI colour codes are stripped before assertions so the checks are stable.
"""

import io
import re

from my_coding_agent import logger as lg

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _capture_stderr(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(lg.sys, "stderr", buf)
    return buf


# --- print_banner ------------------------------------------------------------


def test_print_banner_contains_labels_and_values(monkeypatch):
    buf = _capture_stderr(monkeypatch)
    monkeypatch.setattr(lg, "_git_branch", lambda: "main")
    tools = [{"function": {"name": "bash",
                           "parameters": {"properties": {"command": {}}}}}]
    lg.print_banner(
        label="Coder",
        model="qwen-test",
        tools=tools,
        context_window=131072,
        n_messages=2,
        session_id="abc123",
    )
    out = _plain(buf.getvalue())
    assert "CODER" in out                 # label upper-cased in the title
    assert "qwen-test" in out             # model value
    assert "abc123" in out                # session id
    assert "131,072" in out               # context window formatted with commas
    assert "MODEL" in out and "BRANCH" in out  # info-row labels
    assert "bash(command)" in out         # tool signature row


def test_print_banner_no_tools_omits_tool_section(monkeypatch):
    buf = _capture_stderr(monkeypatch)
    monkeypatch.setattr(lg, "_git_branch", lambda: "main")
    lg.print_banner(label="A", model="m", tools=[], context_window=None)
    out = _plain(buf.getvalue())
    assert "TOOLS" in out          # the metric label still appears
    assert "unknown" in out        # context_window None → "unknown"


def test_print_banner_long_tool_signature_is_truncated(monkeypatch):
    buf = _capture_stderr(monkeypatch)
    monkeypatch.setattr(lg, "_git_branch", lambda: "main")
    props = {f"param_{i}": {} for i in range(40)}
    tools = [{"function": {"name": "huge", "parameters": {"properties": props}}}]
    lg.print_banner(label="A", model="m", tools=tools, context_window=1000)
    out = _plain(buf.getvalue())
    assert "…" in out  # oversized signature is truncated with an ellipsis


# --- _SummaryStyle primitives ------------------------------------------------


def test_summary_style_rows_have_consistent_width():
    s = lg._SummaryStyle()
    rows = [_plain(s.empty_row()), _plain(s.title_row("HELLO")),
            _plain(s.metric_row1("STEPS", "1 / 5"))]
    widths = {len(r) for r in rows}
    assert len(widths) == 1  # every box row is the same visible width


def test_metric_row1_contains_label_and_value():
    s = lg._SummaryStyle()
    assert "STEPS: 1 / 5" in _plain(s.metric_row1("STEPS", "1 / 5"))


# --- _tool_call_rows ---------------------------------------------------------


def test_tool_call_rows_single_line_ok_marker():
    s = lg._SummaryStyle()
    rows = lg._tool_call_rows(s, 1, "bash", {"command": "ls"}, ok=True)
    assert len(rows) == 1
    plain = _plain(rows[0])
    assert "bash(" in plain
    assert "command='ls'" in plain
    assert "✓" in plain


def test_tool_call_rows_failed_marker():
    s = lg._SummaryStyle()
    rows = lg._tool_call_rows(s, 2, "bash", {}, ok=False)
    assert "✗" in _plain(rows[0])


def test_tool_call_rows_skipped_marker():
    s = lg._SummaryStyle()
    rows = lg._tool_call_rows(s, 3, "bash", {}, ok=False, status="skipped")
    assert "⊘" in _plain(rows[0])


def test_tool_call_rows_badges_appear():
    s = lg._SummaryStyle()
    rows = lg._tool_call_rows(s, 1, "bash", {}, ok=True, artifact=True, truncated=True)
    joined = _plain("".join(rows))
    assert "[artifact]" in joined
    assert "[truncated]" in joined


def test_tool_call_rows_long_args_wrap_to_multiple_rows():
    s = lg._SummaryStyle()
    args = {"command": "x" * 300}  # far wider than the box → must wrap
    rows = lg._tool_call_rows(s, 1, "bash", args, ok=True)
    assert len(rows) > 1
    # The status marker only appears on the final row.
    assert "✓" in _plain(rows[-1])
    assert "✓" not in _plain(rows[0])


# --- _tool_calls_section -----------------------------------------------------


def test_tool_calls_section_empty_returns_no_rows():
    s = lg._SummaryStyle()
    assert lg._tool_calls_section(s, []) == []


def test_tool_calls_section_one_row_per_record():
    s = lg._SummaryStyle()
    records = [
        {"name": "bash", "args": {"command": "ls"}, "ok": True},
        {"name": "read_file", "args": {"file_path": "a"}, "ok": False},
    ]
    rows = lg._tool_calls_section(s, records)
    joined = _plain("".join(rows))
    assert "bash(" in joined
    assert "read_file(" in joined


# --- _markdown_rows ----------------------------------------------------------


def test_markdown_rows_renders_text():
    s = lg._SummaryStyle()
    rows = lg._markdown_rows(s, "# Title\n\nSome body text.")
    joined = _plain("".join(rows))
    assert "Title" in joined
    assert "Some body text" in joined


def test_markdown_rows_empty_input_returns_a_row():
    s = lg._SummaryStyle()
    rows = lg._markdown_rows(s, "")
    assert len(rows) >= 1  # never returns an empty list


# --- _token_chart_rows -------------------------------------------------------


def test_token_chart_rows_no_data():
    s = lg._SummaryStyle()
    rows = lg._token_chart_rows(s, None)
    assert "no data" in _plain("".join(rows))


def test_token_chart_rows_no_main_calls():
    s = lg._SummaryStyle()
    internal = [{"kind": "router", "call": 1, "prompt": 1, "completion": 1}]
    rows = lg._token_chart_rows(s, internal)
    assert "no main-agent calls" in _plain("".join(rows))


def test_token_chart_rows_renders_chart_for_main_calls():
    s = lg._SummaryStyle()
    usage = [
        {"kind": "main", "call": 1, "prompt": 100, "completion": 20},
        {"kind": "main", "call": 2, "prompt": 150, "completion": 40},
        {"kind": "router", "call": 3, "prompt": 5, "completion": 2},
    ]
    rows = lg._token_chart_rows(s, usage)
    joined = _plain("".join(rows))
    # The harness-event annotation lists the internal (router) call.
    assert "harness events" in joined
    assert "router" in joined


# --- _handoff_rows / _context_resets_section ---------------------------------


def _handoff(step=2):
    return {
        "step": step,
        "ctx_pct": 80.0,
        "ctx_tokens": 8000,
        "threshold": 75.0,
        "path": "/tmp/handoff.json",
    }


def test_handoff_rows_three_lines_with_values():
    s = lg._SummaryStyle()
    rows = lg._handoff_rows(s, 1, _handoff(), context_window=10000)
    assert len(rows) == 3
    joined = _plain("".join(rows))
    assert "step 2" in joined
    assert "80.0%" in joined
    assert "trigger" in joined
    assert "/tmp/handoff.json" in joined


def test_handoff_rows_truncates_long_path():
    s = lg._SummaryStyle()
    h = _handoff()
    h["path"] = "/very/long/" + "p" * 200 + "/handoff.json"
    rows = lg._handoff_rows(s, 1, h, context_window=10000)
    assert "…" in _plain("".join(rows))


def test_context_resets_section_no_handoffs():
    s = lg._SummaryStyle()
    rows = lg._context_resets_section(s, [], context_window=10000)
    assert "no reset triggered" in _plain("".join(rows))


def test_context_resets_section_with_handoffs():
    s = lg._SummaryStyle()
    rows = lg._context_resets_section(
        s, [_handoff(), _handoff(3)], context_window=10000)
    joined = _plain("".join(rows))
    assert "CONTEXT RESETS" in joined
    assert "step 2" in joined
    assert "step 3" in joined


# --- print_run_summary (integration of the helpers) --------------------------


def test_print_run_summary_renders_full_box(monkeypatch):
    buf = _capture_stderr(monkeypatch)
    monkeypatch.setattr(lg, "_git_branch", lambda: "main")
    lg.print_run_summary(
        steps=3,
        max_steps=5,
        stop_reason="stop",
        prompt_tokens=300,
        completion_tokens=60,
        total_tokens=360,
        context_window=10000,
        elapsed_seconds=12.5,
        tool_records=[{"name": "bash", "args": {"command": "ls"}, "ok": True}],
        handoff_records=[],
        agent_name="Coder",
        last_message="# Done\n\nFinished the task.",
        last_prompt_tokens=2000,
        llm_calls=[{"kind": "main", "call": 1, "prompt": 300, "completion": 60}],
        model="qwen-test",
        session_id="sess1",
        started_at="2026-06-12T00:00:00",
        tools=[{"function": {"name": "bash"}}],
    )
    out = _plain(buf.getvalue())
    assert "CODER  —  RUN COMPLETE" in out
    assert "STEPS: 3 / 5" in out
    assert "STOP REASON: stop" in out
    assert "qwen-test" in out
    assert "TOOL CALLS: 1 (1 ok)" in out
    assert "bash(" in out
    assert "Finished the task" in out
    assert "no reset triggered" in out


def test_print_run_summary_minimal_no_optional_sections(monkeypatch):
    buf = _capture_stderr(monkeypatch)
    monkeypatch.setattr(lg, "_git_branch", lambda: "main")
    lg.print_run_summary(
        steps=1,
        max_steps=1,
        stop_reason="max_steps",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )
    out = _plain(buf.getvalue())
    assert "RUN COMPLETE" in out
    assert "TOOL CALLS: 0" in out
    assert "TOKEN CHART: no data" in out  # llm_calls is None

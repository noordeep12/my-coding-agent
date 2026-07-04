"""Tests for the Agent loop and its helper methods.

The agent now holds an ``LLM`` client via composition (``self.llm``); its
``__init__`` builds that client and attaches a session log. To keep these unit
tests network-free, deterministic, and fast (CONTRIBUTE.md §30), the agent is
built WITHOUT ``__init__`` via ``object.__new__`` — only the attributes the
methods under test actually read are attached, including a bare ``self.llm`` stub
exposing the client state the loop reads (``model``, ``context_window``,
``llm_calls``, hooks, ``chat_completion``).
"""

import json

import pytest

from my_coding_agent.engine.agent import AgentNode as Agent
from my_coding_agent.engine.llm import LLM
from my_coding_agent.pipeline.context import RunContext
from my_coding_agent.pipeline.nodes.context_guard import ContextGuardNode
from my_coding_agent.pipeline.nodes.finalize_step import FinalizeStepNode
from my_coding_agent.pipeline.nodes.tool_routing import (
    ROUTING_EXCERPT_CHARS,
    ROUTING_SIGNAL_MAX_CHARS,
    _routing_signal,
)

# --- helpers -----------------------------------------------------------------


def _make_agent(silent_logger, **overrides):
    """Construct an Agent without running __init__ (no network, no session log).

    The agent holds an ``LLM`` client (``agent.llm``) via composition; build a
    bare client (``LLM.__init__`` is network-free) and set the client-side state
    the loop reads (``model``, ``context_window``, ``llm_calls``). ``model`` /
    ``context_window`` / ``llm_calls`` overrides are routed onto the held client,
    so callers keep passing them as before.
    """
    agent = object.__new__(Agent)
    agent.logger = silent_logger
    agent.messages = []
    agent.tools = []
    agent.label = "Test Agent"
    agent.context_reset_threshold = 0.75
    agent.last_prompt_tokens = 0
    agent.step_num = 0
    agent.stop_reason = "max_steps"
    agent.tool_records = []
    agent.handoff_records = []
    agent.elapsed_seconds = 0.0
    agent.needs_handback = False
    agent.handback_report = None
    agent.child_rollups = []
    # Held LLM client (composition). LLM.__init__ is network-free; swap in the
    # silent logger and set the state the agent loop reads off the client.
    agent.llm = LLM()
    agent.llm.logger = silent_logger
    agent.llm.model = "test-model"
    agent.llm.context_window = 1000
    agent.llm.llm_calls = []
    # Client-side state historically passed as agent kwargs; route onto the client.
    for client_attr in ("model", "context_window", "llm_calls"):
        if client_attr in overrides:
            setattr(agent.llm, client_attr, overrides.pop(client_attr))
    agent.tool_artifacts = {}  # a per-message ToolExecutor is built inside run()
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


class _Resp:
    """Minimal stand-in for an httpx response carrying a fixed JSON payload."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# --- _routing_signal (now in pipeline.nodes.tool_routing) --------------------


def test_routing_signal_joins_last_user_and_assistant():
    messages = [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "doing it"},
    ]
    assert _routing_signal(messages) == "first task doing it"


def test_routing_signal_uses_most_recent_of_each_role():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ]
    assert _routing_signal(messages) == "new user new assistant"


def test_routing_signal_empty_when_no_messages():
    assert _routing_signal([]) == ""


def test_routing_signal_skips_none_content():
    messages = [
        {"role": "user", "content": None},
        {"role": "assistant", "content": "only this"},
    ]
    assert _routing_signal(messages) == "only this"


def test_routing_signal_tool_call_only_turn_contributes_names():
    messages = [
        {"role": "user", "content": "first task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash"}}],
        },
    ]
    assert _routing_signal(messages) == "first task bash"


def test_routing_signal_changes_with_new_tool_results():
    step1 = [
        {"role": "user", "content": "first task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash"}}],
        },
    ]
    step2 = step1 + [
        {"role": "tool", "name": "bash", "content": "output here"},
    ]
    assert _routing_signal(step1) != _routing_signal(step2)
    assert "output here" in _routing_signal(step2)


def test_routing_signal_caps_oversized_tool_result():
    long_content = "x" * (ROUTING_EXCERPT_CHARS + 500)
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash"}}],
        },
        {"role": "tool", "name": "bash", "content": long_content},
    ]
    signal = _routing_signal(messages)
    assert len(signal) <= ROUTING_SIGNAL_MAX_CHARS
    assert "x" * (ROUTING_EXCERPT_CHARS + 1) not in signal


# --- FinalizeStepNode (previously _track_step_usage on Agent) ---------------


def _make_ctx(llm, messages=None, **kwargs):
    """Minimal RunContext for node unit tests — skips full Agent init."""
    import unittest.mock as mock

    ctx = object.__new__(RunContext)
    ctx.llm = llm
    ctx.messages = messages or []
    ctx.step_num = 1
    ctx.max_steps = 10
    ctx.last_prompt_tokens = 0
    ctx.signal = "CONTINUE"
    ctx.needs_handback = False
    ctx.handback_report = None
    ctx.handoff_content = None
    ctx.recorder = mock.Mock()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


def test_track_step_usage_records_last_prompt_tokens(silent_logger):
    llm = LLM()
    llm.context_window = 1000
    ctx = _make_ctx(llm)
    ctx.last_response = _Resp(
        {"usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}}
    )
    FinalizeStepNode().run(ctx)
    assert ctx.last_prompt_tokens == 120


def test_track_step_usage_missing_usage_defaults_zero(silent_logger):
    llm = LLM()
    llm.context_window = 1000
    ctx = _make_ctx(llm)
    ctx.last_response = _Resp({})
    FinalizeStepNode().run(ctx)
    assert ctx.last_prompt_tokens == 0


def test_track_step_usage_handles_no_context_window(silent_logger):
    llm = LLM()
    llm.context_window = 0
    ctx = _make_ctx(llm)
    ctx.last_response = _Resp({"usage": {"prompt_tokens": 50}})
    FinalizeStepNode().run(ctx)
    assert ctx.last_prompt_tokens == 50


# --- ContextGuardNode (previously _context_preflight on Agent) -----------


def _make_preflight_ctx(context_window, last_prompt_tokens=0, messages=None):
    """Minimal RunContext for ContextGuardNode tests."""
    import unittest.mock as mock

    llm = LLM()
    llm.context_window = context_window
    ctx = object.__new__(RunContext)
    ctx.llm = llm
    ctx.messages = messages or []
    ctx.step_num = 0
    ctx.last_prompt_tokens = last_prompt_tokens
    ctx.context_reset_threshold = 0.75
    ctx.signal = "CONTINUE"
    ctx.stop_reason = "max_steps"
    ctx.continuation_messages = []
    ctx.needs_handback = False
    ctx.handback_report = None
    ctx.handoff_content = None
    ctx.recorder = mock.Mock()
    return ctx


def test_preflight_ok_when_below_all_thresholds():
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=100)
    ContextGuardNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_preflight_ok_when_no_context_window():
    ctx = _make_preflight_ctx(context_window=0)
    ContextGuardNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_preflight_warn_path_still_ok():
    # 0.6 <= pct < threshold(0.75) → warns but proceeds.
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=650)
    ContextGuardNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_preflight_stop_when_context_exhausted():
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=1000)
    ContextGuardNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "context_limit"


def test_preflight_reset_triggers_spawn(mocker):
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=800)
    mocker.patch.object(
        ctx.llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "handoff summary"}}]}),
    )
    cont = [{"role": "assistant", "content": "done"}]
    node = ContextGuardNode(spawn_fn=lambda: cont)
    node.run(ctx)
    assert ctx.signal == "RESET"
    assert ctx.continuation_messages == cont
    # The handoff summary is produced in-pipeline by ContextSummarizerNode
    # (triggered by the guard) before the continuation is spawned.
    assert ctx.handoff_content == "handoff summary"
    ctx.recorder.record_summarizer.assert_called_once()
    assert (
        ctx.recorder.record_summarizer.call_args.kwargs["triggered_by"]
        == "context_guard"
    )


def test_preflight_context_limit_synthesizes_handback(mocker):
    """A context-limit stop on a delegated run synthesizes the hand-back report."""
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=1000)
    ctx.needs_handback = True
    mocker.patch.object(
        ctx.llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "synth report"}}]}),
    )
    ContextGuardNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "context_limit"
    assert ctx.handback_report == "synth report"
    kwargs = ctx.recorder.record_summarizer.call_args.kwargs
    assert kwargs["kind"] == "report"
    assert kwargs["triggered_by"] == "context_guard"


def test_preflight_context_limit_standalone_skips_synthesis(mocker):
    """A standalone run pays no report synthesis on a context-limit stop."""
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=1000)
    chat = mocker.patch.object(ctx.llm, "chat_completion")
    ContextGuardNode().run(ctx)
    assert ctx.signal == "STOP"
    chat.assert_not_called()
    ctx.recorder.record_summarizer.assert_not_called()


def test_preflight_reset_without_spawn_fn_stops():
    # When no spawn_fn is provided, a reset threshold hit becomes a STOP.
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=800)
    ContextGuardNode(spawn_fn=None).run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "context_limit"


def test_preflight_estimates_tokens_when_no_last_prompt():
    # last_prompt_tokens == 0 → falls back to a char estimate of json(messages)//2.
    big = "x" * 4000
    ctx = _make_preflight_ctx(
        context_window=1000,
        last_prompt_tokens=0,
        messages=[{"role": "user", "content": big}],
    )
    # len(json.dumps(messages)) // 2 is well over 1000 → hard stop.
    ContextGuardNode().run(ctx)
    assert ctx.signal == "STOP"


# --- step-ceiling cutoff (FinalizeStepNode → ContextSummarizerNode) -----------


def _make_cutoff_resp(finish_reason="tool_calls"):
    return _Resp({"choices": [{"finish_reason": finish_reason}], "usage": {}})


def test_finalize_step_ceiling_sets_explicit_stop(mocker):
    """The final permitted step with the model still going is an explicit cutoff."""
    llm = LLM()
    llm.context_window = 1000
    chat = mocker.patch.object(llm, "chat_completion")
    ctx = _make_ctx(llm)
    ctx.step_num = ctx.max_steps
    ctx.last_response = _make_cutoff_resp()
    FinalizeStepNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "max_steps"
    # Standalone run: no hand-back owed, so no synthesis is paid.
    chat.assert_not_called()
    ctx.recorder.record_summarizer.assert_not_called()


def test_finalize_step_ceiling_synthesizes_handback(mocker):
    """A delegated run's cutoff triggers report synthesis from FinalizeStepNode."""
    llm = LLM()
    llm.context_window = 1000
    mocker.patch.object(
        llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "synth report"}}]}),
    )
    ctx = _make_ctx(llm)
    ctx.step_num = ctx.max_steps
    ctx.needs_handback = True
    ctx.last_response = _make_cutoff_resp()
    FinalizeStepNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "max_steps"
    assert ctx.handback_report == "synth report"
    kwargs = ctx.recorder.record_summarizer.call_args.kwargs
    assert kwargs["kind"] == "report"
    assert kwargs["triggered_by"] == "finalize_step"


def test_finalize_clean_stop_on_final_step_skips_synthesis(mocker):
    """A clean finish on the last permitted step is not a cutoff."""
    llm = LLM()
    llm.context_window = 1000
    chat = mocker.patch.object(llm, "chat_completion")
    ctx = _make_ctx(llm)
    ctx.step_num = ctx.max_steps
    ctx.needs_handback = True
    ctx.last_response = _make_cutoff_resp(finish_reason="stop")
    FinalizeStepNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "stop"
    chat.assert_not_called()
    ctx.recorder.record_summarizer.assert_not_called()


# --- run loop ----------------------------------------------------------------


def _stub_run_internals(agent, mocker):
    """Neutralize the I/O-heavy parts of run() so the loop logic can be tested."""
    # ToolRouter is now instantiated per step inside ToolRoutingNode; patch at the
    # class level so every instance's route_tools passes all tools through.
    mocker.patch(
        "my_coding_agent.pipeline.nodes.tool_routing.ToolRouter.route_tools",
        return_value=[],
    )
    mocker.patch.object(agent, "_save_session_data")
    mocker.patch.object(agent, "_print_summary")
    # Banner is emitted at the start of run() (not __init__) — stub its output.
    mocker.patch("my_coding_agent.engine.agent.print_banner")
    mocker.patch("my_coding_agent.engine.agent.detach_session_log")
    # session_data.json existence check in finally → pretend it already exists so
    # the finally block does not try to save/summarize/detach again.
    mocker.patch("my_coding_agent.engine.agent.Path.exists", return_value=True)
    agent._session_log_handler = (None, None, None)
    agent.session_id = "testsession"
    # Observability recorder is read by run() (start/finish/record_handoff); stub
    # it so the loop logic is exercised without writing events.jsonl.
    agent.recorder = mocker.Mock()
    agent.llm._recorder = agent.recorder


def test_run_stops_on_finish_reason_stop(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    resp = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "all done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    mocker.patch.object(agent.llm, "chat_completion", return_value=resp)
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=5)
    assert agent.stop_reason == "stop"


def test_main_agent_execute_records_no_report_node(silent_logger, mocker):
    """execute() must not emit a report node; that is only for delegated subagents."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    resp = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "all done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    mocker.patch.object(agent.llm, "chat_completion", return_value=resp)
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=5)
    agent.recorder.record_report.assert_not_called()


def test_run_stops_at_max_steps(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    resp = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "still going"},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    mocker.patch.object(agent.llm, "chat_completion", return_value=resp)
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=1)
    assert agent.stop_reason == "max_steps"


@pytest.mark.parametrize("max_steps", [1, 3, 20])
def test_run_executes_exactly_max_steps(silent_logger, mocker, max_steps):
    """Regression (G-05): a run that never finishes early performs EXACTLY
    max_steps main chat_completion calls — not max_steps + 1 — and reports a
    step count equal to max_steps with stop_reason "max_steps".
    """
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    resp = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "working"},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    chat = mocker.patch.object(agent.llm, "chat_completion", return_value=resp)
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=max_steps)

    assert chat.call_count == max_steps
    assert agent.step_num == max_steps
    assert agent.stop_reason == "max_steps"


def test_run_skips_step_on_empty_message(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    empty = _Resp({"choices": [], "usage": {}})
    stop = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    chat = mocker.patch.object(agent.llm, "chat_completion", side_effect=[empty, stop])
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=5)
    # First call returned empty (step skipped), second returned a stop.
    assert chat.call_count == 2
    assert agent.stop_reason == "stop"


def test_run_returns_continuation_result_on_reset(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    cont = [{"role": "assistant", "content": "continuation finished"}]

    # Patch ContextGuardNode.run to immediately signal RESET with the
    # continuation result — this replaces patching the old _context_preflight.
    def _fake_preflight(ctx):
        ctx.signal = "RESET"
        ctx.continuation_messages = cont

    mocker.patch(
        "my_coding_agent.pipeline.nodes.context_guard.ContextGuardNode.run",
        side_effect=_fake_preflight,
    )
    mocker.patch.object(agent.llm, "chat_completion")  # must not be reached

    result = agent.execute(max_steps=5)
    assert result == cont
    agent.llm.chat_completion.assert_not_called()


# --- _generate_handoff -------------------------------------------------------


def test_generate_handoff_builds_and_saves(silent_logger, mocker):
    agent = _make_agent(
        silent_logger,
        messages=[{"role": "user", "content": "task"}],
        context_window=10000,
    )
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "handoff summary"}}]}),
    )
    saved = mocker.patch(
        "my_coding_agent.engine.agent.save_handoff",
        return_value="/tmp/h.json",
    )
    handoff = agent._generate_handoff(step_num=2, prompt_tokens=8000)
    assert handoff.content == "handoff summary"
    assert handoff.step_num == 2
    saved.assert_called_once()


# --- generate_report ---------------------------------------------------------


def test_generate_report_single_no_tools_call_over_full_conversation(
    silent_logger, mocker
):
    """One tool-free call is made over the whole conversation plus a prompt."""
    convo = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "find X"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "X is at line 42"},
    ]
    agent = _make_agent(silent_logger, messages=convo)
    agent.recorder = mocker.Mock()
    chat = mocker.patch.object(
        agent.llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "final report"}}]}),
    )

    result = agent.generate_report()

    assert result == "final report"
    chat.assert_called_once()
    sent_messages = chat.call_args[0][0]
    _, kwargs = chat.call_args
    assert kwargs["tools"] == []
    assert kwargs["kind"] == "report"
    # The full conversation is preserved, with exactly one report prompt appended,
    # so the final tool call and its result are included in the summarized input.
    assert sent_messages[: len(convo)] == convo
    assert len(sent_messages) == len(convo) + 1
    assert sent_messages[-1]["role"] == "user"


def test_generate_report_records_distinct_report_node(silent_logger, mocker):
    agent = _make_agent(silent_logger, messages=[{"role": "user", "content": "t"}])
    agent.recorder = mocker.Mock()
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "report body"}}]}),
    )

    agent.generate_report()

    agent.recorder.record_report.assert_called_once_with("report body")


def test_generate_report_falls_back_when_empty(silent_logger, mocker):
    agent = _make_agent(silent_logger, messages=[{"role": "user", "content": "t"}])
    agent.recorder = mocker.Mock()
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": ""}}]}),
    )

    result = agent.generate_report()

    assert result == "(subagent produced no report)"
    agent.recorder.record_report.assert_called_once_with(
        "(subagent produced no report)"
    )


def test_generate_report_uses_reasoning_when_content_empty(silent_logger, mocker):
    """Reasoning models (Qwen3-thinking) end the summary turn with a tool call,
    leaving ``content`` empty while the report sits in ``reasoning_content``.

    Regression for session d31be8c8c224, where three subagents returned
    "(subagent produced no report)" because the real summary was discarded from
    ``reasoning_content``. The substance must be used, not the placeholder.
    """
    agent = _make_agent(silent_logger, messages=[{"role": "user", "content": "t"}])
    agent.recorder = mocker.Mock()
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        return_value=_Resp(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "I fetched the data: 1631 CVEs.",
                            "tool_calls": [{"id": "1", "function": {"name": "bash"}}],
                        }
                    }
                ]
            }
        ),
    )

    result = agent.generate_report()

    assert result == "I fetched the data: 1631 CVEs."
    agent.recorder.record_report.assert_called_once_with(
        "I fetched the data: 1631 CVEs."
    )


# --- final_assistant_text ------------------------------------------------------


def test_final_assistant_text_returns_content(silent_logger):
    agent = _make_agent(
        silent_logger,
        messages=[
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "the report"},
            {"role": "tool", "content": "ignored"},
        ],
    )
    assert agent.final_assistant_text() == "the report"


def test_final_assistant_text_falls_back_to_reasoning(silent_logger):
    agent = _make_agent(
        silent_logger,
        messages=[
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reasoned report",
            }
        ],
    )
    assert agent.final_assistant_text() == "reasoned report"


def test_final_assistant_text_no_assistant_message(silent_logger):
    agent = _make_agent(silent_logger, messages=[{"role": "user", "content": "t"}])
    assert agent.final_assistant_text() == ""


def test_final_assistant_text_both_fields_empty(silent_logger):
    agent = _make_agent(
        silent_logger,
        messages=[{"role": "assistant", "content": None, "reasoning_content": ""}],
    )
    assert agent.final_assistant_text() == ""


# --- _print_summary ----------------------------------------------------------


def test_print_summary_forwards_aggregates(silent_logger, mocker):
    agent = _make_agent(
        silent_logger,
        messages=[{"role": "assistant", "content": "final answer"}],
        context_window=10000,
        last_prompt_tokens=2000,
        llm_calls=[{"prompt": 100, "completion": 20, "total": 120}],
        model="m",
        session_id="s1",
        started_at="2026-06-12",
    )
    spy = mocker.patch("my_coding_agent.engine.agent.print_run_summary")
    agent._print_summary(max_steps=5)
    kwargs = spy.call_args.kwargs
    assert kwargs["prompt_tokens"] == 100
    assert kwargs["completion_tokens"] == 20
    assert kwargs["last_message"] == "final answer"


# --- _save_session_data ------------------------------------------------------


def test_save_session_data_writes_json(silent_logger, mocker, tmp_path):
    agent = _make_agent(
        silent_logger,
        messages=[{"role": "assistant", "content": "done"}],
        context_window=10000,
        last_prompt_tokens=0,
        llm_calls=[{"prompt": 10, "completion": 5, "total": 15}],
        model="m",
        session_id="sess9",
        started_at="2026-06-12",
    )
    mocker.patch(
        "my_coding_agent.engine.agent.Path",
        lambda *a: tmp_path.joinpath(*a),
    )
    agent._save_session_data(max_steps=5)
    out = tmp_path / ".my_coding_agent" / "sess9" / "session_data.json"
    data = json.loads(out.read_text())
    assert data["session_id"] == "sess9"
    assert data["total_usage"]["total_tokens"] == 15
    assert data["last_message"] == "done"


# --- rollup / usage summary ---------------------------------------------------


def test_usage_summary_no_delegation_equals_own_totals(silent_logger):
    """A run without delegates has rollup grand_total == its own totals."""
    agent = _make_agent(
        silent_logger,
        llm_calls=[
            {"kind": "main", "prompt": 100, "completion": 20, "total": 120},
            {"kind": "main", "prompt": 50, "completion": 10, "total": 60},
        ],
        session_id="solo",
    )
    summary = agent._usage_summary()
    assert summary["descendants"] == []
    assert summary["grand_total"] == {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "total_tokens": 180,
    }
    assert summary["by_kind"]["main"] == {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "total_tokens": 180,
    }


def test_usage_summary_decomposes_by_kind(silent_logger):
    agent = _make_agent(
        silent_logger,
        llm_calls=[
            {"kind": "main", "prompt": 100, "completion": 20, "total": 120},
            {"kind": "report", "prompt": 30, "completion": 5, "total": 35},
        ],
        session_id="s1",
    )
    by_kind = agent._usage_summary()["by_kind"]
    assert by_kind["main"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert by_kind["report"] == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }


def test_add_child_usage_folds_into_grand_total(silent_logger):
    """Parent rollup grand total == sum of every LLM call of every kind across
    the parent and its descendants, including transitively via a child's own
    rollup (D3)."""
    agent = _make_agent(
        silent_logger,
        llm_calls=[{"kind": "main", "prompt": 10, "completion": 5, "total": 15}],
        session_id="parent",
    )
    grandchild_summary = {
        "session_id": "grandchild",
        "elapsed_s": 1.0,
        "steps": 1,
        "by_kind": {
            "main": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        },
        "descendants": [],
        "grand_total": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        },
    }
    child_summary = {
        "session_id": "child",
        "elapsed_s": 2.0,
        "steps": 2,
        "by_kind": {
            "main": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
        },
        "descendants": [grandchild_summary],
        "grand_total": {
            "prompt_tokens": 25,
            "completion_tokens": 5,
            "total_tokens": 30,
        },
    }
    agent.add_child_usage(child_summary)
    summary = agent._usage_summary()
    assert summary["descendants"] == [child_summary]
    # own (10/5/15) + child's grand_total (25/5/30) = 35/10/45
    assert summary["grand_total"] == {
        "prompt_tokens": 35,
        "completion_tokens": 10,
        "total_tokens": 45,
    }


def test_save_session_data_includes_rollup(silent_logger, mocker, tmp_path):
    agent = _make_agent(
        silent_logger,
        messages=[{"role": "assistant", "content": "done"}],
        llm_calls=[{"kind": "main", "prompt": 10, "completion": 5, "total": 15}],
        session_id="sess10",
        started_at="2026-06-12",
    )
    mocker.patch(
        "my_coding_agent.engine.agent.Path",
        lambda *a: tmp_path.joinpath(*a),
    )
    agent._save_session_data(max_steps=5)
    out = tmp_path / ".my_coding_agent" / "sess10" / "session_data.json"
    data = json.loads(out.read_text())
    assert data["rollup"]["grand_total"]["total_tokens"] == 15
    assert data["rollup"]["descendants"] == []


# --- _spawn_continuation -----------------------------------------------------


def test_spawn_continuation_seeds_system_plus_handoff(silent_logger, mocker):
    agent = _make_agent(
        silent_logger,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ],
        model="m",
        context_reset_threshold=0.75,
    )
    # The continuation is built from client-side state read off the held client.
    agent.llm.api_url = "http://x"
    agent.llm.api_key = "k"
    handoff = mocker.Mock()
    mocker.patch(
        "my_coding_agent.engine.agent.handoff_to_user_message",
        return_value={"role": "user", "content": "HANDOFF"},
    )
    fake_cont = mocker.Mock()
    fake_cont.execute.return_value = [{"role": "assistant", "content": "cont done"}]
    cont_cls = mocker.patch(
        "my_coding_agent.engine.agent.AgentNode", return_value=fake_cont
    )

    result = agent._spawn_continuation(handoff, max_steps=5)
    assert result == [{"role": "assistant", "content": "cont done"}]
    # Continuation is seeded with the system message + the handoff user message.
    seeded = cont_cls.call_args.kwargs["messages"]
    assert seeded[0] == {"role": "system", "content": "sys"}
    assert seeded[1] == {"role": "user", "content": "HANDOFF"}


# --- composition contract (Phase 4) ------------------------------------------


def test_agent_holds_llm_by_composition_not_inheritance(silent_logger):
    """Phase 4: Agent holds an LLM client via composition, not subclassing.

    The agent exposes its client as ``self.llm`` and is no longer an ``LLM``
    instance, so client state (e.g. ``llm_calls``) lives on the held client.
    """
    agent = _make_agent(silent_logger)
    assert isinstance(agent.llm, LLM)
    assert not isinstance(agent, LLM)

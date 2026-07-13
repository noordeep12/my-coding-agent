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
from pathlib import Path

import pytest

from my_coding_agent.pipeline.nodes.agent import AgentNode as Agent
from my_coding_agent.engine.checkpoint import (
    Checkpoint,
    checkpoint_path,
    find_last_resumable,
    save_checkpoint,
)
from my_coding_agent.engine.hooks import Hooks
from my_coding_agent.engine.llm import LLM
from my_coding_agent.engine.llm.errors import LLMHTTPStatusError, LLMTransportError
from my_coding_agent.engine.schema import REPORT_SOURCE_FALLBACK
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
    agent.skills = {}
    agent.loaded_skills = set()
    agent._rendered_index = None
    agent.hooks = Hooks()
    # Resume/resilience state (run-resilience) normally set by __init__.
    agent.resumed_from = None
    agent._resume_step = 0
    agent._resume_prompt_tokens = 0
    agent.failure_error = None
    agent.failure_session_id = None
    agent._continuation = None
    agent._did_context_reset = False
    agent._session_dir = Path(".my_coding_agent") / "testsession"
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

    class _NoopRecorder:
        def resource_rollup(self):
            return None

    agent.recorder = _NoopRecorder()
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


def test_routing_signal_bootstraps_from_last_user_message_only():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "first task"},
    ]
    assert _routing_signal(messages) == "first task"


def test_routing_signal_empty_when_no_messages():
    assert _routing_signal([]) == ""


def test_routing_signal_task_text_leaves_signal_once_assistant_exists():
    messages = [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "doing it"},
    ]
    signal = _routing_signal(messages)
    assert signal == "doing it"
    assert "first task" not in signal


def test_routing_signal_uses_most_recent_assistant_message():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "assistant", "content": "new assistant"},
    ]
    assert _routing_signal(messages) == "new assistant"


def test_routing_signal_skips_none_content():
    messages = [
        {"role": "user", "content": None},
        {"role": "assistant", "content": None},
    ]
    assert _routing_signal(messages) == ""


def test_routing_signal_tool_call_only_turn_contributes_names():
    messages = [
        {"role": "user", "content": "first task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash"}}],
        },
    ]
    assert _routing_signal(messages) == "bash"


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
    ctx.session_id = "test-session"
    ctx.tool_records = []
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


# --- ContextGuardNode supersession pass (issue #121) --------------------


def test_guard_retires_contained_result_and_records_event():
    from my_coding_agent.engine.tool_execution.schema import (
        SUPERSESSION_SIZE_FLOOR_CHARS,
    )
    from my_coding_agent.pipeline.nodes.context_guard import (
        CASE_CONTAINMENT,
        STUB_PREFIX,
    )

    big = "x" * SUPERSESSION_SIZE_FLOOR_CHARS
    ctx = _make_preflight_ctx(
        context_window=1000,
        last_prompt_tokens=100,
        messages=[
            {"role": "tool", "tool_call_id": "call_1", "content": big},
            {"role": "tool", "tool_call_id": "call_2", "content": "pre-" + big},
        ],
    )
    ctx.tool_records = [
        {"name": "bash", "tool_call_id": "call_1", "args": {"cmd": "a"}, "ok": True},
        {"name": "bash", "tool_call_id": "call_2", "args": {"cmd": "b"}, "ok": True},
    ]
    old_message = ctx.messages[0]
    ContextGuardNode().run(ctx)

    assert ctx.messages[0] is not old_message  # replaced, never mutated in place
    assert ctx.messages[0]["role"] == "tool"
    assert ctx.messages[0]["tool_call_id"] == "call_1"
    assert ctx.messages[0]["content"].startswith(STUB_PREFIX)
    assert ctx.messages[1]["content"] == "pre-" + big  # superseding result untouched

    ctx.recorder.record_supersession.assert_called_once_with(
        tool_call_id="call_1",
        tool_name="bash",
        case=CASE_CONTAINMENT,
        superseding_tool_call_id="call_2",
        retired_size=len(big),
        step=0,
    )


def test_guard_supersession_disabled_via_env(monkeypatch):
    from my_coding_agent.engine.tool_execution.schema import (
        SUPERSESSION_SIZE_FLOOR_CHARS,
    )

    monkeypatch.setenv("MCA_SUPERSESSION", "0")
    big = "x" * SUPERSESSION_SIZE_FLOOR_CHARS
    ctx = _make_preflight_ctx(
        context_window=1000,
        last_prompt_tokens=100,
        messages=[
            {"role": "tool", "tool_call_id": "call_1", "content": big},
            {"role": "tool", "tool_call_id": "call_2", "content": "pre-" + big},
        ],
    )
    ctx.tool_records = [
        {"name": "bash", "tool_call_id": "call_1", "args": {"cmd": "a"}, "ok": True},
        {"name": "bash", "tool_call_id": "call_2", "args": {"cmd": "b"}, "ok": True},
    ]
    old_message = ctx.messages[0]
    ContextGuardNode().run(ctx)

    assert ctx.messages[0] is old_message  # untouched — run stays byte-identical
    ctx.recorder.record_supersession.assert_not_called()


def test_guard_no_qualifying_result_no_event():
    ctx = _make_preflight_ctx(
        context_window=1000,
        last_prompt_tokens=100,
        messages=[{"role": "tool", "tool_call_id": "call_1", "content": "small"}],
    )
    ctx.tool_records = [
        {"name": "bash", "tool_call_id": "call_1", "args": {"cmd": "a"}, "ok": True},
    ]
    ContextGuardNode().run(ctx)
    ctx.recorder.record_supersession.assert_not_called()


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
        return_value=([], "empty"),
    )
    mocker.patch.object(agent, "_save_session_data")
    mocker.patch.object(agent, "_print_summary")
    # The per-step checkpoint write (run-resilience D3) touches disk; neutralize
    # it so the loop logic is exercised without writing a checkpoint file.
    mocker.patch.object(agent, "_write_checkpoint")
    # Banner is emitted at the start of run() (not __init__) — stub its output.
    mocker.patch("my_coding_agent.pipeline.nodes.agent.print_banner")
    mocker.patch("my_coding_agent.pipeline.nodes.agent.detach_session_log")
    # session_data.json existence check in finally → pretend it already exists so
    # the finally block does not try to save/summarize/detach again.
    mocker.patch("my_coding_agent.pipeline.nodes.agent.Path.exists", return_value=True)
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


# --- run-level failure handling (run-resilience D6) --------------------------


def test_execute_unrecoverable_llm_failure_sets_classified_stop_reason(
    silent_logger, mocker
):
    """An unrecoverable LLM error ends the run as a first-class stop (no crash):
    a classified stop_reason and the error stashed for the CLI resume hint."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        side_effect=LLMHTTPStatusError("HTTP 400", status_code=400, retryable=False),
    )

    result = agent.execute(max_steps=5)  # returns, does not raise
    assert agent.stop_reason == "llm_failure_http_status"
    assert agent.failure_error is not None
    assert agent.failure_error.classification == "http-status"
    assert result == agent.messages  # state preserved for resume


def test_execute_removes_checkpoint_on_clean_finish(silent_logger, mocker):
    """A cleanly finished run deletes its checkpoint (not resumable)."""
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
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=5)
    assert agent.failure_error is None
    remove.assert_called_once_with(agent._session_dir)


def test_execute_keeps_checkpoint_on_unrecoverable_failure(silent_logger, mocker):
    """An unrecoverable LLM failure keeps the checkpoint so the run is resumable."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        side_effect=LLMHTTPStatusError("HTTP 400", status_code=400, retryable=False),
    )
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=5)
    assert agent.failure_error is not None
    remove.assert_not_called()


def test_execute_keeps_checkpoint_on_max_steps(silent_logger, mocker):
    """A max_steps run keeps its checkpoint — still resumable with a bigger budget."""
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
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=1)
    assert agent.stop_reason == "max_steps"
    assert agent.failure_error is None
    remove.assert_not_called()


def test_execute_drops_own_checkpoint_after_context_reset(silent_logger, mocker):
    """A run that hit a context reset leaves NO own checkpoint, even when the
    continuation propagated an unrecoverable failure: the continuation chain owns
    the resumable checkpoint, so --resume-last targets it, not this run."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    err = LLMHTTPStatusError("HTTP 400", status_code=400, retryable=False)

    class _FakePipeline:
        def execute(self, ctx):
            # Simulate a context reset whose continuation failed unrecoverably.
            agent._did_context_reset = True
            agent.failure_error = err
            agent.failure_session_id = "contsession"
            ctx.stop_reason = "llm_failure_http_status"
            return ctx.messages

    mocker.patch(
        "my_coding_agent.pipeline.build_default_pipeline",
        return_value=_FakePipeline(),
    )
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=5)

    assert agent.failure_error is err
    assert agent.failure_session_id == "contsession"
    # Own checkpoint dropped despite the failure; only the own dir is touched.
    remove.assert_called_once_with(agent._session_dir)


def test_resumed_run_context_reset_clean_finish_clears_source(silent_logger, mocker):
    """A resumed run that hits a context reset with a cleanly-finishing
    continuation drops BOTH its own and the SOURCE checkpoint, so the superseded
    source stops being a --resume-last magnet."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    agent.resumed_from = "sourcesess"

    class _FakePipeline:
        def execute(self, ctx):
            # A context reset whose continuation finished cleanly (no failure).
            agent._did_context_reset = True
            ctx.stop_reason = "stop"
            return ctx.messages

    mocker.patch(
        "my_coding_agent.pipeline.build_default_pipeline",
        return_value=_FakePipeline(),
    )
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=5)

    assert agent.failure_error is None
    remove.assert_any_call(agent._session_dir)
    remove.assert_any_call(Path(".my_coding_agent") / "sourcesess")
    assert remove.call_count == 2


def test_handoff_summary_failure_keeps_main_checkpoint(silent_logger, mocker):
    """A handoff-summary LLM failure BEFORE a continuation is spawned stays a
    normal main-session failure: the reset flag is never set, the own checkpoint
    is kept, and the resume hint points at the main session."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    agent.llm.context_window = 10000
    err = LLMHTTPStatusError("HTTP 400", status_code=400, retryable=False)
    mocker.patch.object(agent, "_generate_handoff", side_effect=err)

    class _FakePipeline:
        def execute(self, ctx):
            # The handoff summary fails before the flag is set or a continuation runs.
            return agent._handle_context_reset(ctx, max_steps=5, t_start=0.0)

    mocker.patch(
        "my_coding_agent.pipeline.build_default_pipeline",
        return_value=_FakePipeline(),
    )
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=5)

    assert agent._did_context_reset is False
    assert agent.failure_error is err
    assert agent.failure_session_id == agent.session_id
    remove.assert_not_called()  # own checkpoint kept (D6 resumable failure)


def test_execute_checkpoints_each_completed_step(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)  # stubs _write_checkpoint as a Mock
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
    mocker.patch.object(agent.llm, "chat_completion", return_value=resp)
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=3)
    # One checkpoint write per completed step.
    assert agent._write_checkpoint.call_count == 3


def test_partial_step_not_checkpointed_on_failure(silent_logger, mocker):
    """A step that raises mid-flight is discarded (D4): no checkpoint for it."""
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        side_effect=LLMTransportError("down"),
    )
    agent.execute(max_steps=5)
    agent._write_checkpoint.assert_not_called()  # step never completed


# --- resume (run-resilience D5) ----------------------------------------------


def _checkpoint(step=4):
    return Checkpoint(
        session_id="deadbeef1234",
        step_num=step,
        last_prompt_tokens=555,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "resume me"},
            {"role": "assistant", "content": "progress so far"},
        ],
    )


def test_from_checkpoint_seeds_resume_state(
    silent_logger, mocker, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # keep .my_coding_agent inside the tmp dir
    cp = _checkpoint(step=4)
    agent = Agent.from_checkpoint(cp, label="Main Agent (resumed)")
    assert agent.resumed_from == "deadbeef1234"  # pragma: allowlist secret
    assert agent._resume_step == 4
    assert agent._resume_prompt_tokens == 555
    assert agent.messages == cp.messages
    assert agent.messages is not cp.messages  # a copy, not the same list
    assert agent.recorder.resumed_from == "deadbeef1234"  # pragma: allowlist secret
    # A brand-new session id, distinct from the dead one it links to.
    assert agent.session_id != "deadbeef1234"  # pragma: allowlist secret


def test_resumed_run_continues_from_next_step(
    silent_logger, mocker, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cp = _checkpoint(step=4)
    agent = Agent.from_checkpoint(cp, label="Main Agent (resumed)")
    agent.llm.logger = silent_logger
    agent.llm.context_window = 8192  # avoid the lazy network probe
    stop = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
    )
    chat = mocker.patch.object(agent.llm, "chat_completion", return_value=stop)

    agent.execute(max_steps=50)
    # First (and only) call continued the counter: step 5, not step 1.
    assert chat.call_count == 1
    assert agent.step_num == 5
    # The resumed session persisted its lineage in session_data.
    data = json.loads(
        (Path(".my_coding_agent") / agent.session_id / "session_data.json").read_text()
    )
    assert data["resumed_from"] == "deadbeef1234"  # pragma: allowlist secret
    # A clean finish leaves no checkpoint, so --resume-last never targets a done
    # run (resume-completed-run finding).
    assert not checkpoint_path(Path(".my_coding_agent") / agent.session_id).exists()


def test_resumed_run_clears_source_checkpoint_on_clean_finish(
    silent_logger, mocker, tmp_path, monkeypatch
):
    """A resumed run that finishes cleanly clears BOTH its own and the source
    session's checkpoint, so --resume-last stops targeting the done source."""
    monkeypatch.chdir(tmp_path)
    cp = _checkpoint(step=4)
    save_checkpoint(Path(".my_coding_agent") / cp.session_id, cp)
    agent = Agent.from_checkpoint(cp, label="Main Agent (resumed)")
    agent.llm.logger = silent_logger
    agent.llm.context_window = 8192
    stop = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
    )
    mocker.patch.object(agent.llm, "chat_completion", return_value=stop)

    agent.execute(max_steps=50)

    assert agent.failure_error is None
    assert not checkpoint_path(Path(".my_coding_agent") / agent.session_id).exists()
    assert not checkpoint_path(Path(".my_coding_agent") / cp.session_id).exists()
    assert find_last_resumable(Path(".my_coding_agent")) is None


def test_resumed_run_keeps_source_checkpoint_on_failure(
    silent_logger, mocker, tmp_path, monkeypatch
):
    """A resumed run that itself fails keeps its own checkpoint and leaves the
    source session's checkpoint in place (both stay resumable)."""
    monkeypatch.chdir(tmp_path)
    cp = _checkpoint(step=4)
    save_checkpoint(Path(".my_coding_agent") / cp.session_id, cp)
    agent = Agent.from_checkpoint(cp, label="Main Agent (resumed)")
    agent.llm.logger = silent_logger
    agent.llm.context_window = 8192
    working = _Resp(
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
    mocker.patch.object(
        agent.llm,
        "chat_completion",
        side_effect=[
            working,
            LLMHTTPStatusError("HTTP 400", status_code=400, retryable=False),
        ],
    )
    _exec = mocker.patch(
        "my_coding_agent.engine.tool_execution.ToolExecutor"
    ).return_value
    _exec.run.return_value = ([], [])
    _exec.tool_artifacts = {}

    agent.execute(max_steps=50)

    assert agent.failure_error is not None
    # The step that completed left an own checkpoint; the failure keeps it.
    assert checkpoint_path(Path(".my_coding_agent") / agent.session_id).exists()
    # The source checkpoint is left untouched (still resumable).
    assert checkpoint_path(Path(".my_coding_agent") / cp.session_id).exists()


def test_resume_past_budget_deletes_neither_checkpoint(
    silent_logger, mocker, tmp_path, monkeypatch
):
    """Resuming a session already at/over the budget runs zero steps and must
    delete NOTHING — the source stays resumable for a larger-budget re-resume."""
    monkeypatch.chdir(tmp_path)
    cp = _checkpoint(step=4)
    save_checkpoint(Path(".my_coding_agent") / cp.session_id, cp)
    agent = Agent.from_checkpoint(cp, label="Main Agent (resumed)")
    agent.llm.logger = silent_logger
    agent.llm.context_window = 8192
    chat = mocker.patch.object(agent.llm, "chat_completion")  # must not be reached
    remove = mocker.patch("my_coding_agent.pipeline.nodes.agent.remove_checkpoint")

    agent.execute(max_steps=4)  # resume_step == max_steps → zero steps run

    chat.assert_not_called()
    assert agent.step_num == 4
    assert agent.stop_reason == "max_steps"
    assert agent.failure_error is None
    remove.assert_not_called()  # neither own nor source checkpoint deleted
    # The source stays targetable for a larger-budget re-resume.
    assert find_last_resumable(Path(".my_coding_agent")) == cp.session_id


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
        "my_coding_agent.pipeline.nodes.agent.save_handoff",
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

    agent.recorder.record_report.assert_called_once_with(
        "report body", source=REPORT_SOURCE_FALLBACK
    )


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
        "(subagent produced no report)", source=REPORT_SOURCE_FALLBACK
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
        "I fetched the data: 1631 CVEs.", source=REPORT_SOURCE_FALLBACK
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
    spy = mocker.patch("my_coding_agent.pipeline.nodes.agent.print_run_summary")
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
        "my_coding_agent.pipeline.nodes.agent.Path",
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
        "my_coding_agent.pipeline.nodes.agent.Path",
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
        "my_coding_agent.pipeline.nodes.agent.handoff_to_user_message",
        return_value={"role": "user", "content": "HANDOFF"},
    )
    fake_cont = mocker.Mock()
    fake_cont.execute.return_value = [{"role": "assistant", "content": "cont done"}]
    cont_cls = mocker.patch(
        "my_coding_agent.pipeline.nodes.agent.AgentNode", return_value=fake_cont
    )

    result = agent._spawn_continuation(handoff, max_steps=5)
    assert result == [{"role": "assistant", "content": "cont done"}]
    # Continuation is seeded with the system message + the handoff user message.
    seeded = cont_cls.call_args.kwargs["messages"]
    assert seeded[0] == {"role": "system", "content": "sys"}
    assert seeded[1] == {"role": "user", "content": "HANDOFF"}


def test_context_reset_continuation_failure_propagates(silent_logger, mocker):
    """An unrecoverable failure inside a post-reset continuation is surfaced to
    the top-level run: failure_error/stop_reason/failure_session_id name the
    continuation's resumable session (D6 across the reset boundary)."""
    agent = _make_agent(
        silent_logger,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ],
        context_window=10000,
    )
    agent.session_id = "toplevel"
    agent.recorder = mocker.Mock()
    agent._session_log_handler = (None, None, None)
    mocker.patch.object(agent, "_save_session_data")
    mocker.patch.object(agent, "_print_summary")
    mocker.patch("my_coding_agent.pipeline.nodes.agent.detach_session_log")
    mocker.patch.object(
        agent,
        "_generate_handoff",
        return_value=mocker.Mock(path="/tmp/h.json", content="summary"),
    )
    mocker.patch(
        "my_coding_agent.pipeline.nodes.agent.handoff_to_user_message",
        return_value={"role": "user", "content": "H"},
    )

    failed_cont = mocker.Mock()
    failed_cont.session_id = "contsession"
    failed_cont.failure_session_id = "contsession"
    failed_cont.stop_reason = "llm_failure_http_status"
    failed_cont.failure_error = LLMHTTPStatusError(
        "HTTP 400", status_code=400, retryable=False
    )
    failed_cont.execute.return_value = [{"role": "assistant", "content": "partial"}]
    mocker.patch("my_coding_agent.pipeline.nodes.agent.AgentNode", return_value=failed_cont)

    ctx = _make_ctx(
        agent.llm,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ],
        step_num=3,
        context_reset_threshold=0.75,
        handoff_records=[],
        tool_records=[],
        tool_artifacts={},
        handback_report=None,
        handoff_content="summary",
    )

    result = agent._handle_context_reset(ctx, max_steps=50, t_start=0.0)

    assert result == [{"role": "assistant", "content": "partial"}]
    assert agent.failure_error is failed_cont.failure_error
    assert ctx.stop_reason == "llm_failure_http_status"
    assert agent.failure_session_id == "contsession"
    # The reset marks this run so execute()'s finally drops its own checkpoint.
    assert agent._did_context_reset is True


# --- composition contract (Phase 4) ------------------------------------------


def test_agent_holds_llm_by_composition_not_inheritance(silent_logger):
    """Phase 4: Agent holds an LLM client via composition, not subclassing.

    The agent exposes its client as ``self.llm`` and is no longer an ``LLM``
    instance, so client state (e.g. ``llm_calls``) lives on the held client.
    """
    agent = _make_agent(silent_logger)
    assert isinstance(agent.llm, LLM)
    assert not isinstance(agent, LLM)

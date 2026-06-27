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
from my_coding_agent.pipeline.nodes.context_preflight import ContextPreflightNode
from my_coding_agent.pipeline.nodes.token_tracking import TokenTrackingNode
from my_coding_agent.pipeline.nodes.tool_routing import _routing_signal

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


# --- TokenTrackingNode (previously _track_step_usage on Agent) ---------------


def _make_ctx(llm, messages=None, **kwargs):
    """Minimal RunContext for node unit tests — skips full Agent init."""
    import unittest.mock as mock

    ctx = object.__new__(RunContext)
    ctx.llm = llm
    ctx.messages = messages or []
    ctx.step_num = 1
    ctx.last_prompt_tokens = 0
    ctx.signal = "CONTINUE"
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
    TokenTrackingNode().run(ctx)
    assert ctx.last_prompt_tokens == 120


def test_track_step_usage_missing_usage_defaults_zero(silent_logger):
    llm = LLM()
    llm.context_window = 1000
    ctx = _make_ctx(llm)
    ctx.last_response = _Resp({})
    TokenTrackingNode().run(ctx)
    assert ctx.last_prompt_tokens == 0


def test_track_step_usage_handles_no_context_window(silent_logger):
    llm = LLM()
    llm.context_window = 0
    ctx = _make_ctx(llm)
    ctx.last_response = _Resp({"usage": {"prompt_tokens": 50}})
    TokenTrackingNode().run(ctx)
    assert ctx.last_prompt_tokens == 50


# --- ContextPreflightNode (previously _context_preflight on Agent) -----------


def _make_preflight_ctx(context_window, last_prompt_tokens=0, messages=None):
    """Minimal RunContext for ContextPreflightNode tests."""
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
    ctx.recorder = mock.Mock()
    return ctx


def test_preflight_ok_when_below_all_thresholds():
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=100)
    ContextPreflightNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_preflight_ok_when_no_context_window():
    ctx = _make_preflight_ctx(context_window=0)
    ContextPreflightNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_preflight_warn_path_still_ok():
    # 0.6 <= pct < threshold(0.75) → warns but proceeds.
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=650)
    ContextPreflightNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_preflight_stop_when_context_exhausted():
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=1000)
    ContextPreflightNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "context_limit"


def test_preflight_reset_triggers_spawn():
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=800)
    cont = [{"role": "assistant", "content": "done"}]
    node = ContextPreflightNode(spawn_fn=lambda: cont)
    node.run(ctx)
    assert ctx.signal == "RESET"
    assert ctx.continuation_messages == cont


def test_preflight_reset_without_spawn_fn_stops():
    # When no spawn_fn is provided, a reset threshold hit becomes a STOP.
    ctx = _make_preflight_ctx(context_window=1000, last_prompt_tokens=800)
    ContextPreflightNode(spawn_fn=None).run(ctx)
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
    ContextPreflightNode().run(ctx)
    assert ctx.signal == "STOP"


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

    # Patch ContextPreflightNode.run to immediately signal RESET with the
    # continuation result — this replaces patching the old _context_preflight.
    def _fake_preflight(ctx):
        ctx.signal = "RESET"
        ctx.continuation_messages = cont

    mocker.patch(
        "my_coding_agent.pipeline.nodes.context_preflight.ContextPreflightNode.run",
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
        "my_coding_agent.engine.agent.ContextHandoff.save",
        return_value="/tmp/h.json",
    )
    handoff = agent._generate_handoff(step_num=2, prompt_tokens=8000)
    assert handoff.content == "handoff summary"
    assert handoff.step_num == 2
    saved.assert_called_once()


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
    handoff.to_user_message.return_value = {"role": "user", "content": "HANDOFF"}
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

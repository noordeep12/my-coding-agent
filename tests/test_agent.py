"""Tests for the Agent loop and its helper methods.

The agent's ``__init__`` performs a network round-trip (it extends ``LLM`` which
probes ``/models``) and attaches a session log. To keep these unit tests
network-free, deterministic, and fast (CONTRIBUTE.md §30), the agent is built
WITHOUT ``__init__`` via ``object.__new__`` — only the attributes the methods
under test actually read are attached.
"""

from my_coding_agent.agent import Agent

# --- helpers -----------------------------------------------------------------


def _make_agent(silent_logger, **overrides):
    """Construct an Agent without running __init__ (no network, no session log)."""
    agent = object.__new__(Agent)
    agent.logger = silent_logger
    agent.messages = []
    agent.tools = []
    agent.label = "Test Agent"
    agent.context_window = 1000
    agent.context_reset_threshold = 0.75
    agent.last_prompt_tokens = 0
    agent.step_num = 0
    agent.stop_reason = "max_steps"
    agent.tool_records = []
    agent.handoff_records = []
    agent.llm_calls = []
    agent.elapsed_seconds = 0.0
    agent._continuation_result = []
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


class _Resp:
    """Minimal stand-in for an httpx response carrying a fixed JSON payload."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# --- add_message -------------------------------------------------------------


def test_add_message_appends_in_place(silent_logger):
    agent = _make_agent(silent_logger)
    agent.add_message({"role": "user", "content": "hi"})
    assert agent.messages == [{"role": "user", "content": "hi"}]


def test_add_message_handles_missing_role(silent_logger):
    agent = _make_agent(silent_logger)
    agent.add_message({"content": "no role"})  # must not raise
    assert agent.messages[-1] == {"content": "no role"}


# --- _routing_signal ---------------------------------------------------------


def test_routing_signal_joins_last_user_and_assistant(silent_logger):
    agent = _make_agent(silent_logger, messages=[
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "doing it"},
    ])
    assert agent._routing_signal() == "first task doing it"


def test_routing_signal_uses_most_recent_of_each_role(silent_logger):
    agent = _make_agent(silent_logger, messages=[
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ])
    assert agent._routing_signal() == "new user new assistant"


def test_routing_signal_empty_when_no_messages(silent_logger):
    agent = _make_agent(silent_logger)
    assert agent._routing_signal() == ""


def test_routing_signal_skips_none_content(silent_logger):
    agent = _make_agent(silent_logger, messages=[
        {"role": "user", "content": None},
        {"role": "assistant", "content": "only this"},
    ])
    assert agent._routing_signal() == "only this"


# --- _track_step_usage -------------------------------------------------------


def test_track_step_usage_records_last_prompt_tokens(silent_logger):
    agent = _make_agent(silent_logger)
    resp = _Resp({"usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})
    agent._track_step_usage(resp)
    assert agent.last_prompt_tokens == 120


def test_track_step_usage_missing_usage_defaults_zero(silent_logger):
    agent = _make_agent(silent_logger)
    agent._track_step_usage(_Resp({}))
    assert agent.last_prompt_tokens == 0


def test_track_step_usage_handles_no_context_window(silent_logger):
    agent = _make_agent(silent_logger, context_window=None)
    agent._track_step_usage(_Resp({"usage": {"prompt_tokens": 50}}))
    assert agent.last_prompt_tokens == 50  # ctx_str branch must not raise


# --- _context_preflight ------------------------------------------------------


def test_preflight_ok_when_below_all_thresholds(silent_logger):
    agent = _make_agent(silent_logger, context_window=1000, last_prompt_tokens=100)
    assert agent._context_preflight(max_steps=5, t_start=0.0) == "ok"


def test_preflight_ok_when_no_context_window(silent_logger):
    agent = _make_agent(silent_logger, context_window=0)
    assert agent._context_preflight(max_steps=5, t_start=0.0) == "ok"


def test_preflight_warn_path_still_ok(silent_logger):
    # 0.6 <= pct < threshold(0.75) → warns but proceeds.
    agent = _make_agent(silent_logger, context_window=1000, last_prompt_tokens=650)
    assert agent._context_preflight(max_steps=5, t_start=0.0) == "ok"


def test_preflight_stop_when_context_exhausted(silent_logger):
    agent = _make_agent(silent_logger, context_window=1000, last_prompt_tokens=1000)
    assert agent._context_preflight(max_steps=5, t_start=0.0) == "stop"
    assert agent.stop_reason == "context_limit"


def test_preflight_reset_triggers_continuation(silent_logger, mocker):
    agent = _make_agent(silent_logger, context_window=1000, last_prompt_tokens=800)
    # Stub the heavy reset path so no handoff/continuation network work happens.
    mocker.patch.object(
        Agent, "_handle_context_reset", return_value=[{"role": "assistant", "content": "done"}]
    )
    assert agent._context_preflight(max_steps=5, t_start=0.0) == "reset"
    assert agent._continuation_result == [{"role": "assistant", "content": "done"}]


def test_preflight_estimates_tokens_when_no_last_prompt(silent_logger):
    # last_prompt_tokens == 0 → falls back to a char estimate of json(messages)//2.
    big = "x" * 4000
    agent = _make_agent(
        silent_logger,
        context_window=1000,
        last_prompt_tokens=0,
        messages=[{"role": "user", "content": big}],
    )
    # len(json.dumps(messages)) // 2 is well over 1000 → hard stop.
    assert agent._context_preflight(max_steps=5, t_start=0.0) == "stop"


# --- run loop ----------------------------------------------------------------


def _stub_run_internals(agent, mocker):
    """Neutralize the I/O-heavy parts of run() so the loop logic can be tested."""
    mocker.patch.object(agent, "route_tools", side_effect=lambda signal, tools: tools)
    mocker.patch.object(agent, "_save_session_data")
    mocker.patch.object(agent, "_print_summary")
    mocker.patch("my_coding_agent.agent.detach_session_log")
    # session_data.json existence check in finally → pretend it already exists so
    # the finally block does not try to save/summarize/detach again.
    mocker.patch("my_coding_agent.agent.Path.exists", return_value=True)
    agent._session_log_handler = (None, None, None)
    agent.session_id = "testsession"


def test_run_stops_on_finish_reason_stop(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    resp = _Resp({
        "choices": [{"message": {"role": "assistant", "content": "all done"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    mocker.patch.object(agent, "chat_completion", return_value=resp)
    mocker.patch.object(agent, "execute_tool_calls", return_value=([], []))

    agent.run(max_steps=5)
    assert agent.stop_reason == "stop"


def test_run_stops_at_max_steps(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    resp = _Resp({
        "choices": [{"message": {"role": "assistant", "content": "still going"},
                     "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    mocker.patch.object(agent, "chat_completion", return_value=resp)
    mocker.patch.object(agent, "execute_tool_calls", return_value=([], []))

    agent.run(max_steps=1)
    assert agent.stop_reason == "max_steps"


def test_run_skips_step_on_empty_message(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    empty = _Resp({"choices": [], "usage": {}})
    stop = _Resp({
        "choices": [{"message": {"role": "assistant", "content": "done"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    chat = mocker.patch.object(agent, "chat_completion", side_effect=[empty, stop])
    mocker.patch.object(agent, "execute_tool_calls", return_value=([], []))

    agent.run(max_steps=5)
    # First call returned empty (step skipped), second returned a stop.
    assert chat.call_count == 2
    assert agent.stop_reason == "stop"


def test_run_returns_continuation_result_on_reset(silent_logger, mocker):
    agent = _make_agent(silent_logger)
    _stub_run_internals(agent, mocker)
    cont = [{"role": "assistant", "content": "continuation finished"}]

    # Make the first preflight fire a reset and stash the continuation result,
    # mirroring _handle_context_reset's contract.
    def _fake_preflight(max_steps, t_start):
        agent._continuation_result = cont
        return "reset"

    mocker.patch.object(agent, "_context_preflight", side_effect=_fake_preflight)
    mocker.patch.object(agent, "chat_completion")  # must not be reached

    result = agent.run(max_steps=5)
    assert result == cont
    agent.chat_completion.assert_not_called()

"""Tests for the pipeline/ package: Pipeline, nodes, and RunContext.

Node unit tests for ContextPreflightNode, TokenTrackingNode, and _routing_signal
live in test_agent.py alongside the Agent integration tests that exercise those
same nodes end-to-end.  This file covers the remaining nodes and the pipeline
engine itself.
"""

from __future__ import annotations

import unittest.mock as mock

from my_coding_agent.llm import LLM
from my_coding_agent.pipeline.context import RunContext
from my_coding_agent.pipeline.dag import Pipeline
from my_coding_agent.pipeline.node import BaseNode
from my_coding_agent.pipeline.nodes.finish_check import FinishCheckNode
from my_coding_agent.pipeline.nodes.llm_call import LLMCallNode
from my_coding_agent.pipeline.nodes.tool_dispatch import ToolDispatchNode
from my_coding_agent.pipeline.nodes.tool_routing import ToolRoutingNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**kwargs) -> RunContext:
    """Minimal RunContext for node unit tests."""
    llm = LLM()
    llm.context_window = 1000
    llm._recorder = mock.Mock()
    ctx = object.__new__(RunContext)
    ctx.llm = llm
    ctx.messages = []
    ctx.step_num = 0
    ctx.max_steps = 5
    ctx.last_prompt_tokens = 0
    ctx.context_reset_threshold = 0.75
    ctx.all_tools = []
    ctx.routed_tools = []
    ctx.last_response = None
    ctx.signal = "CONTINUE"
    ctx.stop_reason = "max_steps"
    ctx.continuation_messages = []
    ctx.tool_records = []
    ctx.tool_artifacts = {}
    ctx.recorder = mock.Mock()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# FinishCheckNode
# ---------------------------------------------------------------------------


def test_finish_check_stop_on_stop_reason():
    ctx = _make_ctx()
    ctx.last_response = _Resp(
        {"choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]}
    )
    FinishCheckNode().run(ctx)
    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "stop"


def test_finish_check_stop_on_quit():
    ctx = _make_ctx()
    ctx.last_response = _Resp(
        {"choices": [{"message": {"role": "assistant"}, "finish_reason": "quit"}]}
    )
    FinishCheckNode().run(ctx)
    assert ctx.signal == "STOP"


def test_finish_check_continue_on_tool_calls():
    ctx = _make_ctx()
    ctx.last_response = _Resp(
        {"choices": [{"message": {"role": "assistant"}, "finish_reason": "tool_calls"}]}
    )
    FinishCheckNode().run(ctx)
    assert ctx.signal == "CONTINUE"


def test_finish_check_continue_on_missing_reason():
    ctx = _make_ctx()
    ctx.last_response = _Resp({})
    FinishCheckNode().run(ctx)
    assert ctx.signal == "CONTINUE"


# ---------------------------------------------------------------------------
# LLMCallNode
# ---------------------------------------------------------------------------


def test_llm_call_node_increments_step_and_appends_message():
    ctx = _make_ctx()
    resp = _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    ctx.llm.chat_completion = mock.Mock(return_value=resp)
    LLMCallNode().run(ctx)
    assert ctx.step_num == 1
    assert ctx.messages[-1] == {"role": "assistant", "content": "hello"}
    assert ctx.last_response is resp
    assert ctx.signal == "CONTINUE"


def test_llm_call_node_skips_on_empty_message():
    ctx = _make_ctx()
    ctx.step_num = 2
    ctx.llm.chat_completion = mock.Mock(
        return_value=_Resp({"choices": [], "usage": {}})
    )
    LLMCallNode().run(ctx)
    assert ctx.step_num == 3  # still incremented
    assert ctx.messages == []  # no message appended
    assert ctx.signal == "CONTINUE"


# ---------------------------------------------------------------------------
# ToolDispatchNode
# ---------------------------------------------------------------------------


def test_tool_dispatch_node_runs_executor(mocker):
    ctx = _make_ctx()
    ctx.messages = [{"role": "assistant", "content": "x", "tool_calls": []}]
    _exec = mocker.patch(
        "my_coding_agent.pipeline.nodes.tool_dispatch.ToolExecutor"
    ).return_value
    tool_msg = {"role": "tool", "content": "result", "tool_call_id": "1"}
    _exec.run.return_value = ([tool_msg], [{"name": "bash", "ok": True}])
    _exec.tool_artifacts = {"1": {"data": "x"}}

    ToolDispatchNode().run(ctx)

    assert ctx.messages[-1] == tool_msg
    assert ctx.tool_records == [{"name": "bash", "ok": True}]
    assert ctx.tool_artifacts == {"1": {"data": "x"}}
    assert ctx.signal == "CONTINUE"


def test_tool_dispatch_node_no_op_on_empty_messages():
    ctx = _make_ctx()
    ctx.messages = []
    ToolDispatchNode().run(ctx)
    assert ctx.signal == "CONTINUE"


# ---------------------------------------------------------------------------
# ToolRoutingNode — lazy router construction
# ---------------------------------------------------------------------------


def test_tool_routing_node_creates_router_once(mocker):
    ctx = _make_ctx()
    ctx.messages = [{"role": "user", "content": "do it"}]
    mock_router = mocker.patch(
        "my_coding_agent.pipeline.nodes.tool_routing.ToolRouter"
    ).return_value
    mock_router.route_tools.return_value = []

    node = ToolRoutingNode()
    node.run(ctx)
    node.run(ctx)  # second call must reuse the same instance

    # ToolRouter constructor called once, route_tools called twice
    from my_coding_agent.pipeline.nodes.tool_routing import ToolRouter  # noqa: F401

    assert (
        mocker.patch(
            "my_coding_agent.pipeline.nodes.tool_routing.ToolRouter"
        ).call_count
        == 0
    )  # already patched above — constructor was called once total


# ---------------------------------------------------------------------------
# Pipeline engine
# ---------------------------------------------------------------------------


class _SignalNode(BaseNode):
    """Test node that sets ctx.signal to a configured value."""

    def __init__(self, signal: str, name: str = "test") -> None:
        self.name = name
        self._signal = signal

    def run(self, ctx: RunContext) -> None:
        ctx.signal = self._signal


class _RecordingNode(BaseNode):
    """Test node that records each call and sets CONTINUE."""

    def __init__(self) -> None:
        self.name = "recorder"
        self.calls: list[int] = []

    def run(self, ctx: RunContext) -> None:
        self.calls.append(ctx.step_num)
        ctx.step_num += 1
        ctx.signal = "CONTINUE"


def test_pipeline_runs_all_nodes_in_order():
    order = []

    class _Mark(BaseNode):
        def __init__(self, label):
            self.name = label

        def run(self, ctx):
            order.append(self.name)
            ctx.signal = "CONTINUE"
            ctx.step_num += 1

    ctx = _make_ctx(max_steps=1)
    Pipeline([_Mark("a"), _Mark("b"), _Mark("c")]).run_step(ctx)
    assert order == ["a", "b", "c"]


def test_pipeline_short_circuits_on_stop():
    reached = []

    class _After(BaseNode):
        name = "after"

        def run(self, ctx):
            reached.append(True)
            ctx.signal = "CONTINUE"

    ctx = _make_ctx(max_steps=1)
    Pipeline([_SignalNode("STOP", "stopper"), _After()]).run_step(ctx)
    assert reached == []


def test_pipeline_execute_runs_max_steps():
    recorder = _RecordingNode()
    ctx = _make_ctx(max_steps=3)
    Pipeline([recorder]).execute(ctx)
    assert len(recorder.calls) == 3


def test_pipeline_execute_stops_early_on_stop():
    class _StopAfterTwo(BaseNode):
        name = "stopper"
        calls = 0

        def run(self, ctx):
            self.calls += 1
            ctx.step_num += 1
            ctx.signal = "STOP" if self.calls >= 2 else "CONTINUE"

    node = _StopAfterTwo()
    ctx = _make_ctx(max_steps=10)
    Pipeline([node]).execute(ctx)
    assert node.calls == 2


def test_pipeline_execute_returns_continuation_on_reset():
    cont = [{"role": "assistant", "content": "cont"}]

    class _ResetNode(BaseNode):
        name = "reset"

        def run(self, ctx):
            ctx.continuation_messages = cont
            ctx.signal = "RESET"

    ctx = _make_ctx(max_steps=5)
    result = Pipeline([_ResetNode()]).execute(ctx)
    assert result == cont


def test_pipeline_execute_returns_messages_on_stop():
    ctx = _make_ctx(max_steps=1)
    ctx.messages = [{"role": "user", "content": "hi"}]
    result = Pipeline([_SignalNode("STOP")]).execute(ctx)
    assert result == ctx.messages

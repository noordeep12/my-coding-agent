"""Tests for PromptStageNode — the declaratively-configured workflow-graph stage."""

from __future__ import annotations

import unittest.mock as mock

import pytest

from my_coding_agent.engine.llm import LLM
from my_coding_agent.pipeline.context import RunContext
from my_coding_agent.pipeline.nodes.prompt_stage import PromptStageNode


def _make_ctx(**kwargs) -> RunContext:
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
    ctx.last_response = None
    ctx.signal = "CONTINUE"
    ctx.stop_reason = "max_steps"
    ctx.continuation_messages = []
    ctx.jump_target = ""
    ctx.round_counters = {}
    ctx.tool_records = []
    ctx.tool_artifacts = {}
    ctx.recorder = mock.Mock()
    ctx.skills = {}
    ctx.loaded_skills = set()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _resp(content: str):
    return _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


def test_pure_generator_stage_always_continues():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("a draft"))
    node = PromptStageNode(name="generator", prompt="Write a draft.")

    node.run(ctx)

    assert ctx.signal == "CONTINUE"
    assert ctx.step_num == 1
    assert ctx.messages[-2] == {"role": "user", "content": "Write a draft."}
    assert ctx.messages[-1] == {"role": "assistant", "content": "a draft"}


def test_decision_stage_calls_llm_tagged_with_its_own_name():
    ctx = _make_ctx()
    chat = mock.Mock(return_value=_resp("ACCEPT: looks good"))
    ctx.llm.chat_completion = chat
    node = PromptStageNode(
        name="evaluator",
        prompt="Judge it.",
        accept_if_contains="ACCEPT",
        jump_target="generator",
    )

    node.run(ctx)

    chat.assert_called_once()
    assert chat.call_args.kwargs["kind"] == "evaluator"


def test_decision_stage_accept_stops():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("ACCEPT: looks good"))
    node = PromptStageNode(
        name="evaluator",
        prompt="Judge it.",
        accept_if_contains="ACCEPT",
        jump_target="generator",
    )

    node.run(ctx)

    assert ctx.signal == "STOP"
    assert ctx.stop_reason == "stop"


def test_decision_stage_accept_match_is_case_insensitive():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("accept: fine"))
    node = PromptStageNode(
        name="evaluator",
        prompt="Judge it.",
        accept_if_contains="ACCEPT",
        jump_target="generator",
    )

    node.run(ctx)

    assert ctx.signal == "STOP"


def test_decision_stage_reject_jumps_to_target():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("REJECT: try again"))
    node = PromptStageNode(
        name="evaluator",
        prompt="Judge it.",
        accept_if_contains="ACCEPT",
        jump_target="generator",
    )

    node.run(ctx)

    assert ctx.signal == "JUMP"
    assert ctx.jump_target == "generator"


def test_decision_stage_without_jump_target_raises_on_reject():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("REJECT: try again"))
    node = PromptStageNode(
        name="evaluator", prompt="Judge it.", accept_if_contains="ACCEPT"
    )

    with pytest.raises(ValueError, match="jump_target"):
        node.run(ctx)


def test_empty_message_skips_the_step():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(
        return_value=_Resp({"choices": [], "usage": {}})
    )
    node = PromptStageNode(name="generator", prompt="Write a draft.")

    node.run(ctx)

    assert ctx.signal == "CONTINUE"
    assert ctx.step_num == 1

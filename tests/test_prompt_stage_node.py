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
    ctx.node_threads = {}
    ctx.node_outputs = {}
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
    # The node's own private thread carries the full turn...
    assert ctx.node_threads["generator"][-2] == {
        "role": "user",
        "content": "Write a draft.",
    }
    assert ctx.node_threads["generator"][-1] == {
        "role": "assistant",
        "content": "a draft",
    }
    # ...while the shared audit trail only mirrors the reply.
    assert ctx.messages[-1] == {"role": "assistant", "content": "a draft"}
    assert ctx.node_outputs["generator"] == "a draft"


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


# ---------------------------------------------------------------------------
# Per-node isolation (issue #228 D-isolation)
# ---------------------------------------------------------------------------


def test_own_system_prompt_and_seed_task_seed_only_this_nodes_thread():
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("a draft"))
    node = PromptStageNode(
        name="generator",
        prompt="Write it now.",
        system_prompt="You are GENERATOR.",
        seed_task="The overall task.",
    )

    node.run(ctx)

    thread = ctx.node_threads["generator"]
    assert thread[0] == {"role": "system", "content": "You are GENERATOR."}
    assert thread[1] == {"role": "user", "content": "The overall task."}
    assert thread[2] == {"role": "user", "content": "Write it now."}


def test_system_prompt_and_seed_task_only_seed_once():
    """A second round must not re-append the system prompt / seed task."""
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("draft"))
    node = PromptStageNode(
        name="generator",
        prompt="Write it now.",
        system_prompt="You are GENERATOR.",
        seed_task="The overall task.",
    )

    node.run(ctx)
    node.run(ctx)

    thread = ctx.node_threads["generator"]
    assert sum(1 for m in thread if m.get("role") == "system") == 1
    assert (
        sum(1 for m in thread if m == {"role": "user", "content": "The overall task."})
        == 1
    )


def test_receives_from_injects_only_the_other_nodes_output_text():
    """The evaluator sees the generator's draft text — not its system prompt,
    its own private thread, or any other internal state.
    """
    ctx = _make_ctx()
    ctx.node_outputs["generator"] = "roses are red"
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("ACCEPT: fine"))
    node = PromptStageNode(
        name="evaluator",
        prompt="Judge it.",
        system_prompt="You are EVALUATOR.",
        receives_from="generator",
        accept_if_contains="ACCEPT",
        jump_target="generator",
    )

    node.run(ctx)

    thread = ctx.node_threads["evaluator"]
    assert thread[0] == {"role": "system", "content": "You are EVALUATOR."}
    assert thread[1] == {"role": "user", "content": "roses are red"}
    assert thread[2] == {"role": "user", "content": "Judge it."}
    # Nothing from the generator's own thread/system prompt leaked in.
    assert not any("GENERATOR" in (m.get("content") or "") for m in thread)


def test_receives_from_with_no_prior_output_is_a_no_op():
    """On a node's very first call, the source it receives from may not have
    produced anything yet — no placeholder message should be injected.
    """
    ctx = _make_ctx()
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("draft"))
    node = PromptStageNode(
        name="generator", prompt="Write it now.", receives_from="evaluator"
    )

    node.run(ctx)

    thread = ctx.node_threads["generator"]
    assert thread[0] == {"role": "user", "content": "Write it now."}


def test_generator_never_sees_evaluators_own_reasoning():
    """Only the evaluator's plain output text crosses over — its own thread
    (any chain-of-thought / reasoning content) never does.
    """
    ctx = _make_ctx()
    ctx.node_outputs["evaluator"] = "REJECT: needs more syllables"
    ctx.node_threads["evaluator"] = [
        {"role": "system", "content": "You are EVALUATOR."},
        {"role": "user", "content": "secret internal reasoning about syllables"},
        {"role": "assistant", "content": "REJECT: needs more syllables"},
    ]
    ctx.llm.chat_completion = mock.Mock(return_value=_resp("revised draft"))
    node = PromptStageNode(
        name="generator", prompt="Revise it.", receives_from="evaluator"
    )

    node.run(ctx)

    thread = ctx.node_threads["generator"]
    assert thread[0] == {"role": "user", "content": "REJECT: needs more syllables"}
    assert not any(
        "secret internal reasoning" in (m.get("content") or "") for m in thread
    )

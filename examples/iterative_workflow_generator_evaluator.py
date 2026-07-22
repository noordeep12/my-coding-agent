"""Example: a bounded generator/evaluator loop using declared pipeline transitions.

Demonstrates the iterative-workflow-control capability (issue #228): a two-node
pipeline where the evaluator jumps back to the generator until it accepts, or
until the declared per-loop round ceiling is reached. Uses a scripted (mocked)
LLM client so it runs with no live server.

Run it with:

    uv run python examples/iterative_workflow_generator_evaluator.py

It prints the run's session id and stop reason, and leaves a real session
under .my_coding_agent/<session_id>/ whose events.jsonl contains one
`transition` event per rejected round — open it in the Trace Explorer
(`my-coding-agent-webui --dir .my_coding_agent`) to see the transition nodes
rendered with their round/outcome badges.
"""

from __future__ import annotations

import unittest.mock as mock

from my_coding_agent.pipeline import BaseNode, Pipeline, RunContext
from my_coding_agent.pipeline.nodes.agent import AgentNode
from my_coding_agent.pipeline.schema import Transition


class _Resp:
    """Minimal stand-in for the httpx response the real LLM client returns."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class GeneratorNode(BaseNode):
    """Drafts (or redrafts) the answer, then hands off to the evaluator."""

    name = "generator"

    def run(self, ctx: RunContext) -> None:
        """Call the LLM for a new draft and append it to the conversation."""
        ctx.step_num += 1
        resp = ctx.llm.chat_completion(ctx.messages, tools=ctx.all_tools)
        ctx.last_response = resp
        ctx.messages.append(resp.json()["choices"][0]["message"])
        ctx.signal = "CONTINUE"


class EvaluatorNode(BaseNode):
    """Accepts the draft, or jumps back to the generator for another round."""

    name = "evaluator"

    def run(self, ctx: RunContext) -> None:
        """Judge the latest draft: accept and stop, or jump back for another round."""
        ctx.step_num += 1
        resp = ctx.llm.chat_completion(ctx.messages, tools=ctx.all_tools)
        ctx.last_response = resp
        message = resp.json()["choices"][0]["message"]
        ctx.messages.append(message)
        if "ACCEPT" in (message.get("content") or ""):
            ctx.stop_reason = "stop"
            ctx.signal = "STOP"
        else:
            ctx.signal = "JUMP"
            ctx.jump_target = "generator"


def _build_pipeline(spawn_fn=None, checkpoint_fn=None) -> Pipeline:  # noqa: ARG001
    """Two-node pipeline with one bounded backward transition.

    ``spawn_fn`` is accepted (unused) only to match ``build_default_pipeline``'s
    signature, since this replaces it via ``mock.patch`` below. ``max_rounds=5``
    is the loop's hard ceiling — a fifth rejection would stop the run with
    ``stop_reason = "loop_bound:evaluator->generator"`` instead of jumping again.
    """
    return Pipeline(
        [GeneratorNode(), EvaluatorNode()],
        checkpoint_fn=checkpoint_fn,
        transitions=[Transition(source="evaluator", target="generator", max_rounds=5)],
    )


def _resp(content: str) -> _Resp:
    return _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
    )


def main() -> None:
    """Run the scripted generator/evaluator loop and print the outcome."""
    agent = AgentNode(
        messages=[{"role": "user", "content": "Write a haiku about testing."}],
        tools=[],
        label="GenEvalExample",
    )

    # Scripted responses: two rejected drafts, then an accepted third.
    responses = [
        _resp("draft v1: roses are red"),
        _resp("REJECT: not a haiku"),
        _resp("draft v2: bugs hide in code"),
        _resp("REJECT: still not 5-7-5"),
        _resp(
            "draft v3: tests pass at dawn / "
            "silent code finds its own peace / green light, no more bugs"
        ),
        _resp("ACCEPT: good haiku"),
    ]
    agent.llm.chat_completion = mock.Mock(side_effect=responses)

    with mock.patch("my_coding_agent.pipeline.build_default_pipeline", _build_pipeline):
        agent.execute(max_steps=20)

    print("session_id:", agent.session_id)
    print("stop_reason:", agent.stop_reason)
    print(f"events: .my_coding_agent/{agent.session_id}/events.jsonl")


if __name__ == "__main__":
    main()

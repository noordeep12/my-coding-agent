"""LLMCallNode — call the LLM and append the assistant message."""

from __future__ import annotations

from ...utils import get_logger
from ...utils.parsing import extract_message
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class LLMCallNode(BaseNode):
    """Increment step_num, call chat_completion, and append the response message."""

    name = "llm_call"

    def run(self, ctx: RunContext) -> None:
        ctx.step_num += 1
        _logger.info("--- step %d/%d ---", ctx.step_num, ctx.max_steps)

        resp = ctx.llm.chat_completion(ctx.messages, tools=ctx.routed_tools)
        ctx.last_response = resp

        message = extract_message(resp)
        if not message:
            _logger.error(
                "Step %d: API returned empty message — skipping step", ctx.step_num
            )
            ctx.signal = "CONTINUE"
            return

        ctx.messages.append(message)
        _logger.info(
            "%s message added to the conversation",
            message.get("role", "unknown").upper(),
        )
        ctx.signal = "CONTINUE"

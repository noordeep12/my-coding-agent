"""ToolDispatchNode — execute tool calls from the last assistant message."""

from __future__ import annotations

from ...utils import get_logger
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class ToolDispatchNode(BaseNode):
    """Run ToolExecutor on the last assistant message and merge results into ctx."""

    name = "tool_dispatch"

    def run(self, ctx: RunContext) -> None:
        from ...engine.tool_execution import ToolExecutor  # noqa: PLC0415 (lazy import)

        if not ctx.messages:
            ctx.signal = "CONTINUE"
            return

        last_message = ctx.messages[-1]
        executor = ToolExecutor(
            last_message,
            ctx.llm,
            tools=ctx.all_tools,
            skills=ctx.skills,
            loaded_skills=ctx.loaded_skills,
            step_num=ctx.step_num,
        )
        tool_messages, records = executor.run()

        ctx.tool_artifacts.update(executor.tool_artifacts)
        ctx.tool_records.extend(records)

        for tool_message in tool_messages or []:
            ctx.messages.append(tool_message)
            _logger.info(
                "%s message added to the conversation",
                tool_message.get("role", "unknown").upper(),
            )

        ctx.signal = "CONTINUE"

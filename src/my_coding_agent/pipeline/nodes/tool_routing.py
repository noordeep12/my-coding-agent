"""ToolRoutingNode — select the relevant tool subset for this step."""

from __future__ import annotations

from ..context import RunContext
from ..node import BaseNode
from .router import ToolRouter


class ToolRoutingNode(BaseNode):
    """Route ctx.all_tools down to a relevant subset and store in ctx.routed_tools."""

    name = "tool_routing"

    def __init__(self) -> None:
        self._router: ToolRouter | None = None

    def run(self, ctx: RunContext) -> None:
        if self._router is None:
            self._router = ToolRouter(ctx.llm)
        signal = _routing_signal(ctx.messages)
        ctx.routed_tools = self._router.route_tools(signal, ctx.all_tools)
        ctx.signal = "CONTINUE"


def _routing_signal(messages: list[dict]) -> str:
    """Combine the last user and assistant messages into the router signal."""
    last_user = next(
        (
            m.get("content", "") or ""
            for m in reversed(messages)
            if m.get("role") == "user"
        ),
        "",
    )
    last_assistant = next(
        (
            m.get("content", "") or ""
            for m in reversed(messages)
            if m.get("role") == "assistant"
        ),
        "",
    )
    return " ".join(filter(None, [last_user, last_assistant]))

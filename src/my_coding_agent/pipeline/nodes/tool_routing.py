"""ToolRoutingNode — select the relevant tool subset for this step."""

from __future__ import annotations

from typing import Any

from ...engine.routing import ToolRouter
from ..context import RunContext
from ..node import BaseNode

# Cap on each tool-result excerpt folded into the routing signal, and on the
# total signal length. Routing runs before the LLM call; phase-2 fallback
# embeds the whole signal in a prompt, so both caps keep that call cheap.
ROUTING_EXCERPT_CHARS = 300
ROUTING_SIGNAL_MAX_CHARS = 2000


class ToolRoutingNode(BaseNode):
    """Route ctx.all_tools down to a relevant subset and store in ctx.routed_tools."""

    name = "tool_routing"

    def __init__(self) -> None:
        self._router: ToolRouter | None = None
        self._last_signal: str | None = None
        self._last_selection: list[dict[str, Any]] = []

    def run(self, ctx: RunContext) -> None:
        if self._router is None:
            self._router = ToolRouter(ctx.llm)
        signal = _routing_signal(ctx.messages)
        if signal == self._last_signal:
            ctx.routed_tools = self._last_selection
        else:
            ctx.routed_tools = self._router.route_tools(signal, ctx.all_tools)
            self._last_signal = signal
            self._last_selection = ctx.routed_tools
        ctx.signal = "CONTINUE"


def _routing_signal(messages: list[dict]) -> str:
    """Build the work-state signal: last user message, last assistant message
    (content plus its tool-call function names), and the tool messages that
    followed it (tool name plus a capped excerpt of each result).
    """
    last_user = ""
    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "") or ""
        elif m.get("role") == "assistant":
            last_assistant_idx = i

    parts = [last_user]

    if last_assistant_idx is not None:
        assistant_msg = messages[last_assistant_idx]
        parts.append(assistant_msg.get("content", "") or "")
        tool_call_names = [
            tc.get("function", {}).get("name", "")
            for tc in assistant_msg.get("tool_calls") or []
        ]
        parts.extend(filter(None, tool_call_names))

        for m in messages[last_assistant_idx + 1 :]:
            if m.get("role") != "tool":
                continue
            tool_name = m.get("name", "")
            excerpt = (m.get("content", "") or "")[:ROUTING_EXCERPT_CHARS]
            parts.append(" ".join(filter(None, [tool_name, excerpt])))

    signal = " ".join(filter(None, parts))
    return signal[:ROUTING_SIGNAL_MAX_CHARS]

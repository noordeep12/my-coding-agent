"""ToolRoutingNode — select the relevant tool subset for this step."""

from __future__ import annotations

from typing import Any

from ...engine.routing import _BASELINE_TOOLS, ToolRouter
from ..context import RunContext
from ..node import BaseNode

# Cap on each tool-result excerpt folded into the routing signal, and on the
# total signal length. Routing runs before the LLM call; phase-2 fallback
# embeds the whole signal in a prompt, so both caps keep that call cheap.
ROUTING_EXCERPT_CHARS = 300
ROUTING_SIGNAL_MAX_CHARS = 2000

# A selected non-baseline tool leaves the selection only after this many
# consecutive routings in which none of its tags match the signal.
EXIT_AFTER_MISSES = 2


class ToolRoutingNode(BaseNode):
    """Route ctx.all_tools down to a relevant subset and store in ctx.routed_tools."""

    name = "tool_routing"

    def __init__(self) -> None:
        self._router: ToolRouter | None = None
        self._has_routed = False
        self._selected_names: set[str] = set()
        self._miss_counts: dict[str, int] = {}
        self._previous_recorded_names: frozenset[str] | None = None

    def run(self, ctx: RunContext) -> None:
        if self._router is None:
            self._router = ToolRouter(ctx.llm)
        signal = _routing_signal(ctx.messages)
        raw_selected, phase = self._router.route_tools(
            signal, ctx.all_tools, has_previous_selection=self._has_routed
        )
        selection = self._resolve_selection(ctx.all_tools, raw_selected, phase)
        self._has_routed = True

        names = frozenset(t["function"]["name"] for t in selection)
        if names != self._previous_recorded_names:
            recorder = getattr(ctx.llm, "_recorder", None)
            if recorder is not None:
                recorder.record_router(
                    signal=signal, selected=sorted(names), phase=phase
                )
            self._previous_recorded_names = names

        ctx.routed_tools = selection
        ctx.signal = "CONTINUE"

    def _resolve_selection(
        self,
        all_tools: list[dict[str, Any]],
        raw_selected: list[dict[str, Any]] | None,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Merge this routing's raw result with debounce state into a selection."""
        non_baseline_names = {
            t["function"]["name"]
            for t in all_tools
            if t["function"]["name"] not in _BASELINE_TOOLS
        }

        if phase in ("empty", "no_nonbaseline"):
            self._selected_names = set()
            self._miss_counts = {}
            return raw_selected if raw_selected is not None else []

        if phase == "phase1_baseline":
            matched_now = set(non_baseline_names)
        elif phase == "phase2_llm":
            matched_now = {
                t["function"]["name"] for t in (raw_selected or [])
            } & non_baseline_names
            self._selected_names = set(matched_now)
            self._miss_counts = dict.fromkeys(matched_now, 0)
            return raw_selected if raw_selected is not None else []
        elif phase == "carry_forward":
            matched_now = set()
        else:  # phase1_keyword
            matched_now = {
                t["function"]["name"] for t in (raw_selected or [])
            } & non_baseline_names

        new_selected: set[str] = set(matched_now)
        new_miss: dict[str, int] = dict.fromkeys(matched_now, 0)
        for name in self._selected_names - matched_now:
            misses = self._miss_counts.get(name, 0) + 1
            if misses < EXIT_AFTER_MISSES:
                new_selected.add(name)
                new_miss[name] = misses
        self._selected_names = new_selected
        self._miss_counts = new_miss

        return [
            t
            for t in all_tools
            if t["function"]["name"] in _BASELINE_TOOLS
            or t["function"]["name"] in new_selected
        ]


def _routing_signal(messages: list[dict]) -> str:
    """Build the work-state routing signal.

    Before the first assistant message exists, the signal is the last user
    message content. Once an assistant message exists, the signal is only the
    last assistant message (content plus its tool-call function names) and the
    tool messages that followed it (tool name plus a capped excerpt of each
    result) — the opening/task user text is excluded entirely.
    """
    last_user = ""
    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "") or ""
        elif m.get("role") == "assistant":
            last_assistant_idx = i

    if last_assistant_idx is None:
        return last_user[:ROUTING_SIGNAL_MAX_CHARS]

    assistant_msg = messages[last_assistant_idx]
    parts = [assistant_msg.get("content", "") or ""]
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

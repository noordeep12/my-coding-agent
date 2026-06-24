"""Two-phase tool routing for the pipeline.

Defines ``ToolRouter``, which selects the subset of tools relevant to a message
before each step: phase 1 is a zero-cost keyword match against each tool's tags;
phase 2 is an LLM fallback used only when phase 1 finds no tag matches anywhere.
"""

import json
import re
from typing import TYPE_CHECKING, Any

from ..logger import get_logger
from ..utils import extract_message

if TYPE_CHECKING:
    from ..llm import LLM

_BASELINE_TOOLS = {"bash", "read_file", "read_tool_artifact"}


def _search_bracketed(text: str) -> str:
    """Return the first ``[...]`` span in text, or raise ValueError if absent."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        raise ValueError("no bracketed array found")
    return match.group()


class ToolRouter:
    """Select the relevant tool subset for a message using two-phase routing.

    Phase 1 matches each tool's ``tags`` against the message text (zero cost);
    phase 2 falls back to an LLM call only when no tag matches anywhere.
    Baseline tools (``bash``, ``read_file``, ``read_tool_artifact``) are always
    included.
    """

    def __init__(self, client: "LLM") -> None:
        self.client = client
        self.logger = get_logger(self.__class__.__name__)

    def route_tools(
        self, message: str, all_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return the subset of all_tools relevant to message."""
        if not all_tools:
            return self._finish_route(message, all_tools, "empty")

        text = message.lower()
        baseline = [t for t in all_tools if t["function"]["name"] in _BASELINE_TOOLS]
        non_baseline = [
            t for t in all_tools if t["function"]["name"] not in _BASELINE_TOOLS
        ]

        if not non_baseline:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (no non-baseline tools, skipped)", names
            )
            return self._finish_route(message, all_tools, "no_nonbaseline")

        keyword_matched = [
            t for t in non_baseline if any(tag in text for tag in t.get("tags", []))
        ]

        if keyword_matched:
            selected = baseline + keyword_matched
            names = [t["function"]["name"] for t in selected]
            self.logger.tool("router phase-1 → %s", names)
            return self._finish_route(message, selected, "phase1_keyword")

        baseline_matched = any(
            any(tag in text for tag in t.get("tags", [])) for t in baseline
        )
        if baseline_matched:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (baseline tag match, skipped phase-2)", names
            )
            return self._finish_route(message, all_tools, "phase1_baseline")

        all_names = [t["function"]["name"] for t in all_tools]
        routing_prompt = (
            "You are a tool router. Given the message below, return a JSON "
            f"array of tool names from this list that are relevant: {all_names}.\n"
            "Return only a JSON array, nothing else. "
            "Return [] if no tools are needed.\n\n"
            f"Message: {message}"
        )
        try:
            resp = self.client.chat_completion(
                [{"role": "user", "content": routing_prompt}],
                tools=[],
                kind="tool_router",
            )
            content = extract_message(resp).get("content", "") or ""
            routed_names = None
            for attempt in [
                lambda c: json.loads(c.strip()),
                lambda c: json.loads(_search_bracketed(c)),
                lambda c: json.loads(re.sub(r"```(?:json)?\s*|\s*```", "", c).strip()),
            ]:
                try:
                    routed_names = attempt(content)
                    break
                except Exception:
                    continue
            if routed_names is None:
                raise ValueError(
                    f"could not extract JSON array from: {content[:120]!r}"
                )
        except Exception as exc:
            self.logger.warning("router phase-2 failed (%s), using all tools", exc)
            routed_names = all_names

        valid = {t["function"]["name"] for t in all_tools}
        routed_names = [n for n in routed_names if n in valid]
        selected_names = set(routed_names) | _BASELINE_TOOLS
        selected = [t for t in all_tools if t["function"]["name"] in selected_names]
        self.logger.tool(
            "router phase-2 → %s", [t["function"]["name"] for t in selected]
        )
        return self._finish_route(message, selected, "phase2_llm")

    def _finish_route(
        self, signal: str, selected: list[dict[str, Any]], phase: str
    ) -> list[dict[str, Any]]:
        """Record the selected tool subset and return it."""
        recorder = getattr(self.client, "_recorder", None)
        if recorder is not None:
            recorder.record_router(
                signal=signal,
                selected=[t["function"]["name"] for t in selected],
                phase=phase,
            )
        return selected

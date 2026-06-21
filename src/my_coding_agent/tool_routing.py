"""Two-phase tool routing for the agent loop.

Defines ``ToolRouter``, which selects the subset of tools relevant to a message
before each step: phase 1 is a zero-cost keyword match against each tool's tags;
phase 2 is an LLM fallback used only when phase 1 finds no tag matches anywhere.
The router holds the LLM client (duck-typed) and issues its phase-2 fallback call
through ``client.chat_completion``.
"""

import json
import re
from typing import TYPE_CHECKING, Any

from .logger import get_logger
from .utils import extract_message

if TYPE_CHECKING:
    from .llm import LLM

# Tools always included regardless of routing decision.
_BASELINE_TOOLS = {"bash", "read_file", "read_tool_artifact"}


def _search_bracketed(text: str) -> str:
    """Return the first ``[...]`` span in text, or raise ValueError if absent.

    Used as a JSON-array extraction strategy; raising on no-match lets the caller's
    try/except fall through to the next strategy.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        raise ValueError("no bracketed array found")
    return match.group()


class ToolRouter:
    """Select the relevant tool subset for a message, holding the LLM client.

    Phase 1 matches each tool's ``tags`` against the message text (zero cost);
    phase 2 falls back to an LLM call (``client.chat_completion``) only when no
    tag matches anywhere. Baseline tools (``bash``, ``read_file``,
    ``read_tool_artifact``) are always included.
    """

    def __init__(self, client: "LLM") -> None:
        """Hold the LLM client used for the phase-2 routing fallback.

        Args:
            client: The LLM client whose ``chat_completion`` is used when phase-1
                keyword routing finds no tag matches anywhere.
        """
        self.client = client
        self.logger = get_logger(self.__class__.__name__)

    def route_tools(
        self, message: str, all_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return the subset of all_tools relevant to message.

        Phase 1 — keyword match against each tool's tags (zero cost).
        Phase 2 — LLM fallback only when phase 1 finds zero tag matches
        across ALL tools.
        Baseline tools (bash, read_file, read_tool_artifact) are always included.
        """
        if not all_tools:
            return self._finish_route(message, all_tools, "empty")

        text = message.lower()
        baseline = [t for t in all_tools if t["function"]["name"] in _BASELINE_TOOLS]
        non_baseline = [
            t for t in all_tools if t["function"]["name"] not in _BASELINE_TOOLS
        ]

        # Skip routing entirely when there are no non-baseline tools to choose from.
        if not non_baseline:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (no non-baseline tools, skipped)", names
            )
            return self._finish_route(message, all_tools, "no_nonbaseline")

        # Phase 1: keyword match on tags — check non-baseline tools first.
        keyword_matched = [
            t for t in non_baseline if any(tag in text for tag in t.get("tags", []))
        ]

        if keyword_matched:
            selected = baseline + keyword_matched
            names = [t["function"]["name"] for t in selected]
            self.logger.tool("router phase-1 → %s", names)
            return self._finish_route(message, selected, "phase1_keyword")

        # Phase 1b: check if the message matches any baseline tool's tags.
        # If so, the task clearly needs only baseline tools — skip the LLM call.
        baseline_matched = any(
            any(tag in text for tag in t.get("tags", [])) for t in baseline
        )
        if baseline_matched:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (baseline tag match, skipped phase-2)", names
            )
            return self._finish_route(message, all_tools, "phase1_baseline")

        # Phase 2: LLM fallback — only reached when zero tag matches found anywhere.
        all_names = [t["function"]["name"] for t in all_tools]
        routing_prompt = (
            "You are a tool router. Given the message below, return a JSON "
            "array of tool names "
            f"from this list that are relevant: {all_names}.\n"
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
            # Try multiple extraction strategies in order of reliability.
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

        # Keep baseline + whatever the LLM selected; filter to valid names only
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
        """Record the selected tool subset (if a recorder is attached) and return it."""
        recorder = getattr(self.client, "_recorder", None)
        if recorder is not None:
            recorder.record_router(
                signal=signal,
                selected=[t["function"]["name"] for t in selected],
                phase=phase,
            )
        return selected

"""Two-phase tool routing for the pipeline.

Defines ``ToolRouter``, which selects the subset of tools relevant to a message
before each step: phase 1 is a zero-cost keyword match against each tool's tags;
phase 2 is an LLM fallback used only when phase 1 finds no tag matches anywhere
and there is no previous selection to carry forward instead.
"""

import json
import re
from typing import TYPE_CHECKING, Any

from ..utils import get_logger
from ..utils.parsing import extract_message
from .llm.schema import CALL_KIND_TOOL_ROUTER

if TYPE_CHECKING:
    from .llm import LLM

_BASELINE_TOOLS = {"bash", "read_file", "read_tool_artifact"}


def _search_bracketed(text: str) -> str:
    """Return the first ``[...]`` span in text, or raise ValueError if absent."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        raise ValueError("no bracketed array found")
    return match.group()


def _tag_matches(tag: str, text: str) -> bool:
    """Return True if tag appears in text as a whole word (case-insensitive)."""
    return re.search(rf"\b{re.escape(tag)}\b", text, re.IGNORECASE) is not None


def _any_tag_matches(tool: dict[str, Any], text: str) -> bool:
    return any(_tag_matches(tag, text) for tag in tool.get("tags", []))


class ToolRouter:
    """Select the relevant tool subset for a message using two-phase routing.

    Phase 1 matches each tool's ``tags`` against the message text (zero cost,
    whole-word match); phase 2 falls back to an LLM call only on a cold start
    (no previous selection) with no tag match. A mid-run no-match returns no
    evidence (``None``) so the caller can carry the previous selection forward.
    Baseline tools (``bash``, ``read_file``, ``read_tool_artifact``) are always
    included.
    """

    def __init__(self, client: "LLM") -> None:
        self.client = client
        self.logger = get_logger(self.__class__.__name__)

    def route_tools(
        self,
        message: str,
        all_tools: list[dict[str, Any]],
        has_previous_selection: bool = False,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Return (selected_subset, phase) for message and all_tools.

        selected_subset is ``None`` only when routing finds no evidence
        (``phase == "carry_forward"``) — the caller should reuse its previous
        selection unchanged.
        """
        if not all_tools:
            return all_tools, "empty"

        text = message
        baseline = [t for t in all_tools if t["function"]["name"] in _BASELINE_TOOLS]
        non_baseline = [
            t for t in all_tools if t["function"]["name"] not in _BASELINE_TOOLS
        ]

        if not non_baseline:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (no non-baseline tools, skipped)", names
            )
            return all_tools, "no_nonbaseline"

        keyword_matched = [t for t in non_baseline if _any_tag_matches(t, text)]

        if keyword_matched:
            selected = baseline + keyword_matched
            names = [t["function"]["name"] for t in selected]
            self.logger.tool("router phase-1 → %s", names)
            return selected, "phase1_keyword"

        baseline_matched = any(_any_tag_matches(t, text) for t in baseline)
        if baseline_matched:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (baseline tag match, skipped phase-2)", names
            )
            return all_tools, "phase1_baseline"

        if has_previous_selection:
            self.logger.tool(
                "router phase-1 → no match, carrying forward previous selection"
            )
            return None, "carry_forward"

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
                kind=CALL_KIND_TOOL_ROUTER,
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
        return selected, "phase2_llm"

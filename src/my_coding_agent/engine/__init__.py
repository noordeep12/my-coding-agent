"""Engine package — owns execution: LLM client, tools, and AgentNode."""

from typing import Any

from .llm import LLM, OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL
from .tool_registry import ToolRegistry, function_to_json, tool

__all__ = [
    "AgentNode",
    "DEFAULT_MAX_STEPS",
    "LLM",
    "OMLX_API_URL",
    "OMLX_API_KEY",
    "OMLX_MODEL",
    "ToolRegistry",
    "function_to_json",
    "tool",
]


def __getattr__(name: str) -> Any:
    # AgentNode now lives in pipeline.nodes.agent (issue #203); re-exported
    # here lazily so importing engine submodules doesn't eagerly pull in the
    # pipeline package (which itself depends on engine primitives).
    if name in ("AgentNode", "DEFAULT_MAX_STEPS"):
        from ..pipeline.nodes.agent import DEFAULT_MAX_STEPS, AgentNode

        return {"AgentNode": AgentNode, "DEFAULT_MAX_STEPS": DEFAULT_MAX_STEPS}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

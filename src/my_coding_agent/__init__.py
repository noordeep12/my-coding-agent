"""Public API for my-coding-agent."""

import logging
from importlib.metadata import PackageNotFoundError, version

from .llm import LLM
from .pipeline.nodes.agent_node import AgentNode
from .pipeline.nodes.handoff import ContextHandoff
from .tool_registry import ToolRegistry, tool
from .utils.exceptions import (
    APIResponseError,
    MyCodingAgentError,
    PathTraversalError,
    ToolDefinitionError,
)


def Agent(*args, **kwargs):
    """Deprecated: use AgentNode instead."""
    import warnings

    warnings.warn(
        "Agent is deprecated and will be removed in v0.2.0. Use AgentNode instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return AgentNode(*args, **kwargs)


logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("my-coding-agent")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "LLM",
    "AgentNode",
    "Agent",
    "ContextHandoff",
    "tool",
    "ToolRegistry",
    "MyCodingAgentError",
    "PathTraversalError",
    "ToolDefinitionError",
    "APIResponseError",
]

"""Public API for my-coding-agent."""

import logging
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .engine import LLM, AgentNode, ToolRegistry, tool
from .pipeline.nodes.handoff import ContextHandoff
from .utils.exceptions import (
    APIResponseError,
    MyCodingAgentError,
    PathTraversalError,
    ToolDefinitionError,
)


def Agent(*args: Any, **kwargs: Any) -> "AgentNode":
    """Construct an AgentNode. Deprecated: use AgentNode directly (removed v0.2.0)."""
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

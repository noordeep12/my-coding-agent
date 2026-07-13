"""Public API for my-coding-agent."""

import logging
from importlib.metadata import PackageNotFoundError, version

from .engine import DEFAULT_MAX_STEPS, LLM, AgentNode, ToolRegistry, tool
from .pipeline.schema import ContextHandoff
from .utils.exceptions import (
    APIResponseError,
    MyCodingAgentError,
    PathTraversalError,
    ToolDefinitionError,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("my-coding-agent")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_MAX_STEPS",
    "LLM",
    "APIResponseError",
    "AgentNode",
    "ContextHandoff",
    "MyCodingAgentError",
    "PathTraversalError",
    "ToolDefinitionError",
    "ToolRegistry",
    "__version__",
    "tool",
]

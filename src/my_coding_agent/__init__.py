"""Public API for my-coding-agent.

Re-exports the agent harness's core types — the ``Agent`` loop, the ``LLM``
client, the ``ToolsRegistry`` and ``tool`` converter, the ``ContextHandoff``
state object, and the exception hierarchy.
"""

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent
from .exceptions import (
    APIResponseError,
    MyCodingAgentError,
    PathTraversalError,
    ToolDefinitionError,
)
from .handoff import ContextHandoff
from .llm import LLM
from .tools import ToolsRegistry, tool

try:
    __version__ = version("my-coding-agent")
except PackageNotFoundError:  # not installed (e.g. source tree without install)
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "LLM",
    "Agent",
    "tool",
    "ToolsRegistry",
    "ContextHandoff",
    "MyCodingAgentError",
    "PathTraversalError",
    "ToolDefinitionError",
    "APIResponseError",
]

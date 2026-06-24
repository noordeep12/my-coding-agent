"""Public API for my-coding-agent.

Re-exports the agent harness's core types — the ``Agent`` loop, the ``LLM``
client, the ``ToolRegistry`` and ``tool`` converter, the ``ContextHandoff``
state object, and the exception hierarchy.
"""

import logging
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
from .tool_registry import ToolRegistry, tool

# Library logging convention (CONTRIBUTE.md §31): attach a no-op handler to the
# package's top-level logger once, so the library emits no output and no
# "No handlers could be found" warning unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("my-coding-agent")
except PackageNotFoundError:  # not installed (e.g. source tree without install)
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "LLM",
    "Agent",
    "tool",
    "ToolRegistry",
    "ContextHandoff",
    "MyCodingAgentError",
    "PathTraversalError",
    "ToolDefinitionError",
    "APIResponseError",
]

"""Public API for my-coding-agent.

Re-exports the agent harness's core types — the ``Agent`` loop, the ``LLM``
client, the ``ToolsRegistry`` and ``tool`` converter, the ``ContextHandoff``
state object, and the exception hierarchy.
"""

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

__all__ = [
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

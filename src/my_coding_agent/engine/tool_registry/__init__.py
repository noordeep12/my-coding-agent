"""Tool registry package: tool implementations and OpenAI tool-definition converter.

Public surface:
    ToolRegistry       — callable tools the agent can invoke
    ARTIFACT_THRESHOLD — large-output boundary (chars); bash output above this
                         triggers artifact separation
    artifact_file_path — single source of truth for the per-artifact file path
                         scheme, shared by the executor and the registry
    function_to_json   — convert a Python function to an OpenAI tool definition
    tool               — alias for function_to_json (decorator form)
"""

import logging

from .converter import function_to_json, tool
from .exceptions import ToolDefinitionError
from .registry import ARTIFACT_THRESHOLD, ToolRegistry, artifact_file_path
from .skills import (
    RenderedIndex,
    Skill,
    build_opening_block,
    discover_skills,
    render_skill_index,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ARTIFACT_THRESHOLD",
    "RenderedIndex",
    "Skill",
    "ToolDefinitionError",
    "ToolRegistry",
    "artifact_file_path",
    "build_opening_block",
    "discover_skills",
    "function_to_json",
    "render_skill_index",
    "tool",
]

"""Tool registry package: tool implementations and OpenAI tool-definition converter.

Public surface:
    ToolRegistry       — callable tools the agent can invoke
    ARTIFACT_THRESHOLD — large-output boundary (chars); bash output above this
                         triggers artifact separation
    function_to_json   — convert a Python function to an OpenAI tool definition
    tool               — alias for function_to_json (decorator form)
"""

import logging

from .converter import function_to_json, tool
from .registry import ARTIFACT_THRESHOLD, ToolRegistry

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ToolRegistry",
    "ARTIFACT_THRESHOLD",
    "function_to_json",
    "tool",
]

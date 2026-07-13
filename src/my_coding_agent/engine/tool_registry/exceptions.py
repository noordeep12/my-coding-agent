"""Tool-registry domain exceptions."""

from ...utils.exceptions import MyCodingAgentError


class ToolDefinitionError(MyCodingAgentError, ValueError):
    """Raise when a function cannot be converted into a tool definition."""

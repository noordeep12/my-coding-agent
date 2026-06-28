"""Custom exception hierarchy for my-coding-agent."""


class MyCodingAgentError(Exception):
    """Base exception — catch this to handle all library errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class PathTraversalError(MyCodingAgentError, ValueError):
    """Raise when a resolved path escapes the workspace base directory."""


class ToolDefinitionError(MyCodingAgentError, ValueError):
    """Raise when a function cannot be converted into a tool definition."""


class APIResponseError(MyCodingAgentError, ValueError):
    """Raise when the LLM API returns a response that cannot be parsed."""

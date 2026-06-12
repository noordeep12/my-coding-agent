"""Custom exception hierarchy for my-coding-agent.

Defines a single library-specific base exception so callers can catch all
errors raised by this package with one ``except``, and gives each error an
optional actionable ``hint`` alongside its message (CONTRIBUTE.md §29).
"""


class MyCodingAgentError(Exception):
    """Base exception — catch this to handle all library errors.

    Args:
        message: Human-readable description of what went wrong.
        hint: Optional actionable suggestion for resolving the error.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


# Subclass ValueError as well as the package base: existing callers and the
# agent loop's _RECOVERABLE_EXCEPTIONS treat these as recoverable ValueErrors,
# so the recoverable/fatal classification is unchanged.
class PathTraversalError(MyCodingAgentError, ValueError):
    """Raise when a resolved path escapes the workspace base directory."""


class ToolDefinitionError(MyCodingAgentError, ValueError):
    """Raise when a function cannot be converted into a tool definition."""


class APIResponseError(MyCodingAgentError, ValueError):
    """Raise when the LLM API returns a response that cannot be parsed."""

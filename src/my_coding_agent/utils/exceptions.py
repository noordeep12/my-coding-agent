"""Custom exception hierarchy for my-coding-agent."""

__all__ = [
    "APIResponseError",  # noqa: F822 — resolved lazily via __getattr__ below
    "MyCodingAgentError",
    "PathTraversalError",
    "ToolDefinitionError",  # noqa: F822 — resolved lazily via __getattr__ below
]


class MyCodingAgentError(Exception):
    """Base exception — catch this to handle all library errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class PathTraversalError(MyCodingAgentError, ValueError):
    """Raise when a resolved path escapes the workspace base directory."""


# ToolDefinitionError and APIResponseError relocated to their raising domains
# (engine/tool_registry and engine/llm respectively, per CONTRIBUTE.md §29), both
# of which import MyCodingAgentError from this module. Resolving the re-export
# lazily (PEP 562) avoids the circular import an eager import would create.
def __getattr__(name: str) -> type[MyCodingAgentError]:
    if name == "ToolDefinitionError":
        from ..engine.tool_registry.exceptions import ToolDefinitionError

        return ToolDefinitionError
    if name == "APIResponseError":
        from ..engine.llm.errors import APIResponseError

        return APIResponseError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

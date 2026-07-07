"""Lifecycle hook event names and typed contracts — a stdlib-only leaf module.

Registry/loading/firing logic lives in the sibling ``__init__.py``; this module
holds only event-name constants and dataclasses, per the Schema Convention.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

EVENT_PRE_TOOL_USE = "PreToolUse"
EVENT_POST_TOOL_USE = "PostToolUse"
EVENT_SESSION_START = "SessionStart"
EVENT_SESSION_END = "SessionEnd"

HOOK_EVENTS = (
    EVENT_PRE_TOOL_USE,
    EVENT_POST_TOOL_USE,
    EVENT_SESSION_START,
    EVENT_SESSION_END,
)

# The ``reason`` value used for the no-execution descriptor a ``PreToolUse``
# block resolves to — a fifth sibling of "refused"/"not_found"/"wrong_args"/"raised".
BLOCKED_BY_HOOK_REASON = "blocked_by_hook"


@dataclass(frozen=True)
class HookContext:
    """Immutable snapshot passed to a hook callback at one lifecycle event."""

    event: str
    session_id: str
    step: int = 0
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    result: Any | None = None


@dataclass(frozen=True)
class HookResult:
    """A hook's decision: allow (default) or block with a reason.

    Only meaningful for ``PreToolUse``; a ``PostToolUse``/``SessionStart``/
    ``SessionEnd`` hook's ``blocked`` is ignored by the firing sites.
    """

    blocked: bool = False
    reason: str | None = None

    @classmethod
    def allow(cls) -> "HookResult":
        return cls(blocked=False)

    @classmethod
    def block(cls, reason: str) -> "HookResult":
        return cls(blocked=True, reason=reason)


HookCallback = Callable[[HookContext], "HookResult | None"]


@dataclass(frozen=True)
class HookSpec:
    """One registered hook: which event (+ optional tool matcher) triggers it."""

    name: str
    event: str
    callback: HookCallback
    tool: str | None = None

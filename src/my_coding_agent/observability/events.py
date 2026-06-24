"""Typed event schema for the observability event stream.

Each session writes a newline-delimited ``events.jsonl`` whose rows are the dicts
emitted by :class:`~my_coding_agent.observability.recorder.Recorder`. These
dataclasses are the reader-side model: :mod:`reader` parses the raw rows into
``Session`` / ``LLMCall`` / ``ToolCall`` / ``Handoff`` objects that the HTML
viewer consumes. The recorder writes plain dicts; only the reader uses these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Event ``type`` tags written as the first key of every JSONL row.
SESSION_START = "session_start"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
ROUTER = "router"
HANDOFF = "handoff"
SESSION_END = "session_end"


@dataclass
class LLMCall:
    """One chat-completion call captured at the LLM choke point.

    ``messages`` is the full conversation snapshot at call time — stored only for
    payload-bearing kinds (``main``/``handoff``); ``None`` for ancillary kinds to
    keep the event stream small. ``response`` holds the model's reply
    (``content``/``reasoning``/``tool_calls``).
    """

    call: int
    kind: str
    started_at: str
    latency_s: float
    prompt: int
    completion: int
    total: int
    context_window: int
    messages: list[dict[str, Any]] | None
    response: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMCall:
        """Build an ``LLMCall`` from a raw JSONL row, tolerating missing keys."""
        return cls(
            call=int(d.get("call", 0)),
            kind=str(d.get("kind", "main")),
            started_at=str(d.get("started_at", "")),
            latency_s=float(d.get("latency_s", 0.0)),
            prompt=int(d.get("prompt", 0)),
            completion=int(d.get("completion", 0)),
            total=int(d.get("total", 0)),
            context_window=int(d.get("context_window", 0)),
            messages=d.get("messages"),
            response=d.get("response", {}) or {},
        )

    @property
    def context_pct(self) -> float:
        """Prompt tokens as a percentage of the context window (0 when unknown)."""
        return self.prompt / self.context_window * 100 if self.context_window else 0.0


@dataclass
class ToolCall:
    """One tool dispatch captured via the before/after hooks.

    ``ok``/``status``/``tool_call_id`` are joined in from ``session_data.json`` by
    the reader (the hooks themselves do not see them).
    """

    name: str
    args: dict[str, Any]
    result: str
    latency_s: float
    started_at: str
    child_session_id: str | None = None
    ok: bool | None = None
    status: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        """Build a ``ToolCall`` from a raw JSONL row."""
        return cls(
            name=str(d.get("name", "")),
            args=d.get("args", {}) or {},
            result=str(d.get("result", "")),
            latency_s=float(d.get("latency_s", 0.0)),
            started_at=str(d.get("started_at", "")),
            child_session_id=d.get("child_session_id"),
        )


@dataclass
class Handoff:
    """One context-reset event: the window filled and was summarized/evicted."""

    step: int
    ctx_tokens: int
    ctx_pct: float
    content: str
    path: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Handoff:
        """Build a ``Handoff`` from a raw JSONL row."""
        return cls(
            step=int(d.get("step", 0)),
            ctx_tokens=int(d.get("ctx_tokens", 0)),
            ctx_pct=float(d.get("ctx_pct", 0.0)),
            content=str(d.get("content", "")),
            path=str(d.get("path", "")),
        )


@dataclass
class Session:
    """A parsed agent session: metadata, ordered events, and nested children."""

    session_id: str
    label: str
    model: str
    parent_session_id: str | None
    context_window: int
    started_at: str
    stop_reason: str = "unknown"
    steps: int = 0
    elapsed_s: float = 0.0
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    handoffs: list[Handoff] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    children: list[Session] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the run ended on a clean model-signalled stop."""
        return self.stop_reason in ("stop", "exit", "quit")

    @property
    def total_tokens(self) -> int:
        """Sum of total tokens across every captured LLM call."""
        return sum(c.total for c in self.llm_calls)

    @property
    def prompt_tokens(self) -> int:
        """Sum of prompt tokens across every captured LLM call."""
        return sum(c.prompt for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        """Sum of completion tokens across every captured LLM call."""
        return sum(c.completion for c in self.llm_calls)

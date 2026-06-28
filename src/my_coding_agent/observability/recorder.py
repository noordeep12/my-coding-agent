"""Structured observability recorder — writes a per-session ``events.jsonl``.

The recorder is the capture layer. It emits one JSON object per line as the run
proceeds, so a session that crashes mid-run still leaves a diagnosable trail. It
never touches the ``logger`` package — capture is entirely separate from logging.

Wiring:
- ``record_llm_call`` captures every LLM call with full conversation snapshots.
- ``before_tool`` / ``after_tool`` time and capture each tool's full input/output.
- ``record_handoff`` captures context-reset events.
- ``current_session_id`` lets a delegated subagent link back to its parent.
"""

from __future__ import annotations

import contextvars
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Event type tags written as the first key of every JSONL row ───────────────
SESSION_START = "session_start"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
ROUTER = "router"
HANDOFF = "handoff"
SESSION_END = "session_end"

# Set by ``Agent.run`` for the duration of a run; a child ``Agent`` constructed
# inside ``delegate`` reads it so the session tree can be reconstructed.
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

# The active run's Recorder, so ``delegate`` can attach the spawned child's
# session id to the delegate tool call for an exact parent→child tree link.
current_recorder: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "current_recorder", default=None
)

# Which LLM call kinds store the full ``messages`` snapshot. Every chat-completion
# kind — including the ancillary ones (tool_router, summarizer, arg_correction) —
# keeps its input so the viewer can show each call's input/output like the main
# call. Trim this set to bound the event-stream size if needed.
FULL_PAYLOAD_KINDS: set[str] = {
    "main",
    "handoff",
    "tool_router",
    "tool_output_summarizer",
    "tool_arg_correction",
}


def _now() -> str:
    """Return the current local time as an ISO-8601 string (second precision)."""
    return datetime.now().isoformat(timespec="seconds")


class Recorder:
    """Append-as-you-go writer of a session's structured event stream."""

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        parent_session_id: str | None = None,
    ) -> None:
        """Initialize a recorder for one session (no file I/O until first emit).

        Args:
            session_id: The owning agent's session id.
            session_dir: ``.my_coding_agent/<session_id>/`` — where ``events.jsonl``
                is written, alongside the existing session artifacts.
            parent_session_id: Session id of the spawning agent, or ``None`` for a
                top-level run. Used to rebuild the agent/subagent tree.
        """
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.path = Path(session_dir) / "events.jsonl"
        # Single pending slot: tools dispatch sequentially (before → after), so
        # one in-flight start time is sufficient to compute latency.
        self._pending: tuple[float, str] | None = None
        # Child session id stashed by ``delegate`` and attached to the next
        # ``delegate`` tool-call event for an exact parent→child link.
        self._pending_delegate_child: str | None = None

    def _emit(self, event: dict[str, Any]) -> None:
        """Append one event row to ``events.jsonl``, serialized immediately.

        Serializing synchronously snapshots mutable inputs (e.g. the live
        ``messages`` list) at emit time, and flushing per write keeps the trail
        intact if the process dies. ``default=str`` tolerates odd values.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self, label: str, model: str, context_window: int) -> None:
        """Emit the session-start event with run metadata."""
        self._emit(
            {
                "type": SESSION_START,
                "session_id": self.session_id,
                "parent_session_id": self.parent_session_id,
                "label": label,
                "model": model,
                "context_window": context_window,
                "started_at": _now(),
            }
        )

    def finish(self, stop_reason: str, steps: int, elapsed_s: float) -> None:
        """Emit the session-end event with the final outcome."""
        self._emit(
            {
                "type": SESSION_END,
                "stop_reason": stop_reason,
                "steps": steps,
                "elapsed_s": elapsed_s,
                "ended_at": _now(),
            }
        )

    # ── LLM calls ──────────────────────────────────────────────────────────────
    def record_llm_call(
        self,
        kind: str,
        call: int,
        latency_s: float,
        usage: dict[str, Any],
        messages: list[dict[str, Any]],
        context_window: int,
        response_data: dict[str, Any],
    ) -> None:
        """Record one chat-completion call (full snapshot for payload kinds)."""
        keep_messages = messages if kind in FULL_PAYLOAD_KINDS else None
        self._emit(
            {
                "type": LLM_CALL,
                "call": call,
                "kind": kind,
                "started_at": _now(),
                "latency_s": round(latency_s, 4),
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
                "context_window": context_window,
                "messages": keep_messages,
                "response": _response_summary(response_data),
            }
        )

    # ── tool capture (called directly by the ToolExecutor) ─────────────────────
    def before_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Pre-dispatch hook: stamp the start time. Returns args unchanged."""
        self._pending = (time.monotonic(), name)
        return args

    def after_tool(self, name: str, args: dict[str, Any], result: str) -> str:
        """Post-dispatch hook: emit the tool event with full I/O. Result unchanged."""
        if self._pending is not None and self._pending[1] == name:
            latency = time.monotonic() - self._pending[0]
        else:
            latency = 0.0
        self._pending = None
        event: dict[str, Any] = {
            "type": TOOL_CALL,
            "name": name,
            "args": args,
            "result": result,
            "latency_s": round(latency, 4),
            "started_at": _now(),
        }
        # Attach the spawned subagent's session id to a delegate call so the tree
        # nests the child under this exact tool call.
        if name == "delegate" and self._pending_delegate_child is not None:
            event["child_session_id"] = self._pending_delegate_child
            self._pending_delegate_child = None
        self._emit(event)
        return result

    def note_delegate_child(self, child_session_id: str) -> None:
        """Record the subagent id ``delegate`` just spawned (see ``after_tool``)."""
        self._pending_delegate_child = child_session_id

    # ── tool routing ───────────────────────────────────────────────────────────
    def record_router(self, signal: str, selected: list[str], phase: str) -> None:
        """Record the ToolRouter's selected tool subset for a step."""
        self._emit(
            {
                "type": ROUTER,
                "started_at": _now(),
                "signal": signal[:500],
                "selected": selected,
                "phase": phase,
                "used_llm": phase == "phase2_llm",
            }
        )

    # ── handoff ────────────────────────────────────────────────────────────────
    def record_handoff(
        self, step: int, ctx_tokens: int, ctx_pct: float, content: str, path: str
    ) -> None:
        """Record a context-reset event (window full → summarized/evicted)."""
        self._emit(
            {
                "type": HANDOFF,
                "step": step,
                "ctx_tokens": ctx_tokens,
                "ctx_pct": round(ctx_pct, 1),
                "content": content,
                "path": path,
            }
        )


def _response_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Extract content/reasoning/tool_calls plus the raw assistant message.

    ``raw`` is the full ``choices[0].message`` object as the server returned it,
    so the viewer can show the unmodified LLM response.
    """
    choices = data.get("choices", []) or []
    if not choices:
        return {"content": "", "reasoning": "", "tool_calls": [], "raw": {}}
    msg = choices[0].get("message", {}) or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "raw": msg,
    }

"""Structured observability recorder — writes a per-session ``events.jsonl``.

The recorder is the capture layer. It emits one JSON object per line as the run
proceeds, so a session that crashes mid-run still leaves a diagnosable trail. It
never touches the ``logger`` package — capture is entirely separate from logging.

Wiring:
- ``record_llm_call`` captures each LLM call's messages and tool definitions.
- ``before_tool`` / ``after_tool`` time and capture each tool's full input/output.
- ``record_handoff`` captures context-reset events.
- ``record_report`` captures a subagent's end-of-turn final report.
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
REPORT = "report"
SESSION_END = "session_end"
TOKEN_TRACKING = "token_tracking"
FINISH_CHECK = "finish_check"
SUMMARIZER = "summarizer"

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

# The active run's AgentNode, so ``delegate`` can hand the completed child's
# usage summary straight up to the parent for the persisted rollup (D3) —
# untyped (``Any``) here to avoid a circular import with ``engine.agent``.
current_agent_node: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_agent_node", default=None
)

# Which LLM call kinds store the full input payload — the ``messages`` snapshot
# and the ``tools`` definitions given to the model. Every chat-completion kind —
# including the ancillary ones (tool_router, arg_correction) — keeps
# its input so the viewer can show each call's input/output like the main call.
# Trim this set to bound the event-stream size if needed.
FULL_PAYLOAD_KINDS: set[str] = {
    "main",
    "handoff",
    "report",
    "tool_router",
    "tool_arg_correction",
    "artifact_query",
}


def _now() -> str:
    """Return the current local time as an ISO-8601 string.

    Millisecond precision with an explicit UTC offset: milliseconds keep
    adjacent fast events from colliding on one identical stamp, and the
    offset keeps rows comparable across machine/timezone changes.
    """
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


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
        # one in-flight start time is sufficient to compute latency. Carries
        # both the monotonic start (duration) and the wall-clock start
        # (``started_at``) captured at the same moment.
        self._pending: tuple[float, str, str] | None = None
        # Child session id stashed by ``delegate`` and attached to the next
        # ``delegate`` tool-call event for an exact parent→child link.
        self._pending_delegate_child: str | None = None
        # LLM call numbers made *while* a tool is dispatching (e.g. the bounded
        # extraction call inside ``read_tool_artifact``), attached to the tool's
        # own event so the viewer can nest them under that exact tool call —
        # the same "stash now, attach at after_tool" pattern as the delegate
        # child link above.
        self._pending_child_llm_calls: list[int] = []

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
        tools: list[dict[str, Any]] | None = None,
        started_at: str | None = None,
    ) -> None:
        """Record one chat-completion call (full snapshot for payload kinds).

        For kinds in ``FULL_PAYLOAD_KINDS`` the ``messages`` snapshot and the
        ``tools`` definitions given to the model this turn are both kept, so the
        viewer can show the exact input (conversation + available tools) the
        model saw. Other kinds keep neither, to bound the event-stream size.

        ``started_at`` is the wall-clock moment the call began (captured by the
        caller alongside its monotonic latency timer); falls back to emit time
        when the caller does not supply one.
        """
        if self._pending is not None:
            # A tool is currently dispatching (before_tool ran, after_tool has
            # not) — this call happened inside that tool's own implementation
            # (e.g. read_tool_artifact's extraction call), so it nests under
            # the tool's event rather than the flat session chain.
            self._pending_child_llm_calls.append(call)
        keep_payload = kind in FULL_PAYLOAD_KINDS
        self._emit(
            {
                "type": LLM_CALL,
                "call": call,
                "kind": kind,
                "started_at": started_at or _now(),
                "latency_s": round(latency_s, 4),
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
                "context_window": context_window,
                "messages": messages if keep_payload else None,
                "tools": (tools or []) if keep_payload else None,
                "response": _response_summary(response_data),
            }
        )

    # ── tool capture (called directly by the ToolExecutor) ─────────────────────
    def before_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Pre-dispatch hook: stamp the start time. Returns args unchanged."""
        self._pending = (time.monotonic(), name, _now())
        self._pending_child_llm_calls = []
        return args

    def after_tool(self, name: str, args: dict[str, Any], result: str) -> str:
        """Post-dispatch hook: emit the tool event with full I/O. Result unchanged."""
        if self._pending is not None and self._pending[1] == name:
            latency = time.monotonic() - self._pending[0]
            started_at = self._pending[2]
        else:
            latency = 0.0
            started_at = _now()
        self._pending = None
        event: dict[str, Any] = {
            "type": TOOL_CALL,
            "name": name,
            "args": args,
            "result": result,
            "latency_s": round(latency, 4),
            "started_at": started_at,
        }
        # Attach the spawned subagent's session id to a delegate call so the tree
        # nests the child under this exact tool call.
        if name == "delegate" and self._pending_delegate_child is not None:
            event["child_session_id"] = self._pending_delegate_child
            self._pending_delegate_child = None
        # Attach any LLM calls this tool made internally (e.g. read_tool_artifact's
        # artifact_query extraction) so the tree nests them under this tool call.
        if self._pending_child_llm_calls:
            event["child_llm_calls"] = self._pending_child_llm_calls
            self._pending_child_llm_calls = []
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

    # ── subagent report ─────────────────────────────────────────────────────────
    def record_report(self, content: str) -> None:
        """Record a subagent's end-of-turn final report (distinct from handoff)."""
        self._emit(
            {
                "type": REPORT,
                "content": content,
                "started_at": _now(),
            }
        )

    def record_summarizer(
        self,
        kind: str,
        step: int,
        triggered_by: str,
        latency_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        started_at: str | None = None,
    ) -> None:
        """Record one ContextSummarizerNode invocation, linked to its trigger.

        ``triggered_by`` names the pipeline node that fired the summarizer
        (``finalize_step`` or ``context_guard``) so the viewer can nest the
        summarizer node under it in the trace tree. ``started_at`` is the
        wall-clock moment the summarization began, captured by the caller
        alongside its monotonic latency timer.
        """
        self._emit(
            {
                "type": SUMMARIZER,
                "kind": kind,
                "step": step,
                "triggered_by": triggered_by,
                "latency_s": round(latency_s, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "started_at": started_at or _now(),
            }
        )

    def record_token_tracking(
        self,
        step: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        ctx_pct: float,
        context_window: int,
    ) -> None:
        """Emit token-usage data for a completed pipeline step."""
        self._emit(
            {
                "type": TOKEN_TRACKING,
                "step": step,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "ctx_pct": round(ctx_pct, 1),
                "context_window": context_window,
                "started_at": _now(),
            }
        )

    def record_finish_check(self, step: int, finish_reason: str, signal: str) -> None:
        """Emit the finish-check decision so the trace shows why a step ended."""
        self._emit(
            {
                "type": FINISH_CHECK,
                "step": step,
                "finish_reason": finish_reason,
                "signal": signal,
                "started_at": _now(),
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

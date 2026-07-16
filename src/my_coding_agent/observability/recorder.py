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

from .error_classification import classify_error
from .sampler import get_sampler

# ── Event type tags written as the first key of every JSONL row ───────────────
SESSION_START = "session_start"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
HANDOFF = "handoff"
REPORT = "report"
SESSION_END = "session_end"
TOKEN_TRACKING = "token_tracking"  # noqa: S105 -- event-type label, not a credential
FINISH_CHECK = "finish_check"
SUMMARIZER = "summarizer"
ANOMALY = "anomaly"
REFUSAL = "refusal"
EGRESS = "egress"
SANDBOX_ACTIVATION = "sandbox_activation"
SANDBOX_DENIAL = "sandbox_denial"
PROVENANCE = "provenance"
EXFIL = "exfil"
HOOK = "hook"
SKILL_INDEX = "skill_index"
SUPERSESSION = "supersession"
# Run-resilience (D2): additive events for the LLM outage-recovery loop.
LLM_WAIT = "llm_wait"  # one patient-phase wait before a retry
LLM_RECOVERY = "llm_recovery"  # server answered after a stall
LLM_FAILURE = "llm_failure"  # unrecoverable — tolerance exceeded / non-retryable
TRANSITION = "transition"  # a declared node-to-node JUMP taken or bound-exhausted

# Set by ``Agent.run`` for the duration of a run; a child ``Agent`` constructed
# inside ``delegate`` reads it so the session tree can be reconstructed.
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

# The active run's Recorder, so ``delegate`` can attach the spawned child's
# session id to the delegate tool call for an exact parent→child tree link.
current_recorder: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
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


def now() -> str:
    """Return the current local time as an ISO-8601 string.

    Millisecond precision with an explicit UTC offset: milliseconds keep
    adjacent fast events from colliding on one identical stamp, and the
    offset keeps rows comparable across machine/timezone changes.
    """
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


_now = now


class Recorder:
    """Append-as-you-go writer of a session's structured event stream."""

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        parent_session_id: str | None = None,
        resumed_from: str | None = None,
    ) -> None:
        """Initialize a recorder for one session (no file I/O until first emit).

        Args:
            session_id: The owning agent's session id.
            session_dir: ``.my_coding_agent/<session_id>/`` — where ``events.jsonl``
                is written, alongside the existing session artifacts.
            parent_session_id: Session id of the spawning agent, or ``None`` for a
                top-level run. Used to rebuild the agent/subagent tree.
            resumed_from: Session id this run resumed from a checkpoint (D5), or
                ``None`` for a fresh run. Recorded on ``session_start`` as the
                lineage link, mirroring ``parent_session_id`` for delegation.
        """
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.resumed_from = resumed_from
        self.path = Path(session_dir) / "events.jsonl"
        # Process-wide machine-wide resource sampler (node-resource-monitoring),
        # shared by the main agent and any in-process subagents; start/stop are
        # ref-counted so nested sessions don't tear the thread down under
        # each other.
        self._sampler = get_sampler()
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
        # Per-kind last emitted snapshot, for prefix-delta emission (design D2):
        # the physical call number that carried the snapshot, and the message
        # *objects* themselves (not a copy) so future calls can prove prefix
        # reuse by identity. Relies on the append-or-replace invariant on
        # conversation message lists: a message dict, once appended to a
        # conversation, is never edited in place — only appended after or the
        # whole list is replaced with a new one (e.g. on handoff). This holds
        # everywhere today (audited: pipeline/nodes/llm_call.py,
        # pipeline/nodes/tool_dispatch.py, engine/agent.py,
        # pipeline/nodes/context_summarizer.py all append or replace, never
        # mutate an already-appended dict). If that ever changes, the identity
        # check simply misses the mutation and falls back to a full snapshot —
        # never a wrong delta.
        self._last_snapshot: dict[str, tuple[int, list[dict[str, Any]]]] = {}

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
    def start(self, label: str, model: str, context_window: int, posture: str) -> None:
        """Emit the session-start event with run metadata.

        ``posture`` is the run's protection posture ("sandboxed" or
        "screened_only", see ``engine.tool_execution.policy.
        get_protection_posture``) — recorded once per run, passively; it
        changes no enforcement. Pre-change traces have no ``posture`` key.
        """
        self._sampler.start()
        self._emit(
            {
                "type": SESSION_START,
                "session_id": self.session_id,
                "parent_session_id": self.parent_session_id,
                "resumed_from": self.resumed_from,
                "label": label,
                "model": model,
                "context_window": context_window,
                "posture": posture,
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
        self._sampler.stop()

    def resource_window(self, start: float, end: float) -> dict[str, Any] | None:
        """Summarize machine-wide resource use over a monotonic ``[start, end]``.

        Thin pass-through to the shared sampler, so timed-event call sites
        (LLM call, summarizer) don't need to import the sampler directly.
        Returns ``None`` when capture is unavailable or has been disabled.
        """
        return self._sampler.summarize_window(start, end)

    def resource_rollup(self) -> dict[str, Any] | None:
        """Session-wide resource rollup so far (peaks/averages, byte totals)."""
        return self._sampler.session_rollup()

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
        max_tokens: int | None = None,
        resources: dict[str, Any] | None = None,
    ) -> None:
        """Record one chat-completion call (incremental snapshot for payload kinds).

        For kinds in ``FULL_PAYLOAD_KINDS`` the ``messages`` snapshot and the
        ``tools`` definitions given to the model this turn are both kept, so the
        viewer can show the exact input (conversation + available tools) the
        model saw. Other kinds keep neither, to bound the event-stream size.

        When the current *messages* provably extends the last snapshot
        recorded for this *kind* (identity-verified shared prefix — see D2 in
        the incremental-trace-capture design), only the new suffix is
        emitted, referencing the base call number and prefix length; the
        reader reconstructs the full snapshot at load time. Whenever prefix
        reuse cannot be proven (first call of a kind, a shorter or replaced
        message list, or any identity mismatch in the shared region), the
        full snapshot is emitted, exactly as before — fidelity always wins
        over size.

        ``started_at`` is the wall-clock moment the call began (captured by the
        caller alongside its monotonic latency timer); falls back to emit time
        when the caller does not supply one.

        ``max_tokens`` is the completion token cap in force for the call, if
        any — recorded on the event (absent when uncapped) so the viewer can
        badge a completion that was cut at its cap without importing the
        engine's budget constants (extract-completeness-disclosure D6).

        ``resources`` is the caller's machine-wide sampler-window summary for
        the call's execution bracket (node-resource-monitoring); omitted from
        the event entirely when ``None`` (capture unavailable).
        """
        if self._pending is not None:
            # A tool is currently dispatching (before_tool ran, after_tool has
            # not) — this call happened inside that tool's own implementation
            # (e.g. read_tool_artifact's extraction call), so it nests under
            # the tool's event rather than the flat session chain.
            self._pending_child_llm_calls.append(call)
        keep_payload = kind in FULL_PAYLOAD_KINDS
        event: dict[str, Any] = {
            "type": LLM_CALL,
            "call": call,
            "kind": kind,
            "started_at": started_at or _now(),
            "latency_s": round(latency_s, 4),
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
            "context_window": context_window,
            "tools": (tools or []) if keep_payload else None,
            "response": _response_summary(response_data),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **({"resources": resources} if resources is not None else {}),
        }
        if keep_payload:
            event.update(self._encode_messages(kind, call, messages))
        else:
            event["messages"] = None
        self._emit(event)

    def _encode_messages(
        self, kind: str, call: int, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return the message-payload fields for a payload-kind call.

        Emits a prefix-delta against the last snapshot recorded for *kind*
        when the shared region is identity-verified unchanged; otherwise
        emits the full snapshot. Always updates the per-kind snapshot to
        *(call, messages)* for the next call of this kind to compare against.

        The stored snapshot is a shallow copy of the *list container*
        (``list(messages)``), never the caller's own list object. The caller
        (``ctx.messages``) is appended to in place after this call returns
        (the pipeline appends the assistant reply, then any tool results), so
        holding the live reference would let the "previous" snapshot's length
        silently grow between calls — corrupting ``prefix_len`` for the next
        delta. A shallow copy freezes the length and order at call time while
        the copied elements remain the exact same dict objects, so the
        identity check on the next call is unaffected.
        """
        prev = self._last_snapshot.get(kind)
        fields: dict[str, Any]
        if prev is not None:
            base_call, base_messages = prev
            prefix_len = len(base_messages)
            if len(messages) >= prefix_len and all(
                messages[i] is base_messages[i] for i in range(prefix_len)
            ):
                fields = {
                    "messages": None,
                    "messages_base_call": base_call,
                    "messages_prefix_len": prefix_len,
                    "messages_suffix": messages[prefix_len:],
                }
            else:
                fields = {"messages": messages}
        else:
            fields = {"messages": messages}
        self._last_snapshot[kind] = (call, list(messages))
        return fields

    # ── tool capture (called directly by the ToolExecutor) ─────────────────────
    def before_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Pre-dispatch hook: stamp the start time. Returns args unchanged."""
        self._pending = (time.monotonic(), name, _now())
        self._pending_child_llm_calls = []
        return args

    def after_tool(
        self,
        name: str,
        args: dict[str, Any],
        result: str,
        ok: bool,
        error: str | None,
        timing: tuple[float, float, str] | None = None,
    ) -> str:
        """Post-dispatch hook: emit the tool event with full I/O. Result unchanged.

        ``ok``/``error`` are the outcome the executor already holds (the
        envelope's verdict/error text, or the executor-failure descriptor's
        error text) — the capture-time identity, never re-derived from
        ``result``. On failure, ``error_class`` is computed via the shared
        classification helper so it agrees with the anomaly detector's
        failure signature.

        ``timing`` is passed only by the concurrent dispatch path: a
        ``(start_mono, end_mono, started_at)`` bracket the executor captured
        around this exact call. When present it drives latency/resources
        directly and the single-slot ``_pending`` state is left untouched, so
        overlapping calls (whose before/after can no longer be strictly paired)
        never race it. Concurrent read-only calls make no nested LLM calls, so
        no ``child_llm_calls`` are attached on this path.
        """
        if timing is not None:
            start_mono, end_mono, started_at = timing
            latency = end_mono - start_mono
            resources = self.resource_window(start_mono, end_mono)
        elif self._pending is not None and self._pending[1] == name:
            pending_start = self._pending[0]
            latency = time.monotonic() - pending_start
            started_at = self._pending[2]
            resources = self.resource_window(pending_start, time.monotonic())
            self._pending = None
        else:
            latency = 0.0
            started_at = _now()
            resources = None
            self._pending = None
        event: dict[str, Any] = {
            "type": TOOL_CALL,
            "name": name,
            "args": args,
            "result": result,
            "latency_s": round(latency, 4),
            "started_at": started_at,
            "ok": ok,
        }
        if resources is not None:
            event["resources"] = resources
        if not ok:
            error_text = error or ""
            event["error"] = error_text
            event["error_class"] = classify_error(error_text)
        # Attach the spawned subagent's session id to a delegate call so the tree
        # nests the child under this exact tool call.
        if name == "delegate" and self._pending_delegate_child is not None:
            event["child_session_id"] = self._pending_delegate_child
            self._pending_delegate_child = None
        # Attach any LLM calls this tool made internally (e.g. read_tool_artifact's
        # artifact_query extraction) so the tree nests them under this tool call.
        # The concurrent path (timing given) tracks no such nested calls and must
        # not consume the shared pending list.
        if timing is None and self._pending_child_llm_calls:
            event["child_llm_calls"] = self._pending_child_llm_calls
            self._pending_child_llm_calls = []
        self._emit(event)
        return result

    def note_delegate_child(self, child_session_id: str) -> None:
        """Record the subagent id ``delegate`` just spawned (see ``after_tool``)."""
        self._pending_delegate_child = child_session_id

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
    def record_report(self, content: str, source: str) -> None:
        """Record a subagent's end-of-turn final report (distinct from handoff).

        ``source`` names which of the three cost-divergent paths produced the
        report (``verbatim`` / ``summarizer`` / ``fallback``, from
        ``engine.schema``) — required, no default, so a future call site
        cannot record a report without declaring its path.
        """
        self._emit(
            {
                "type": REPORT,
                "content": content,
                "source": source,
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
        resources: dict[str, Any] | None = None,
    ) -> None:
        """Record one ContextSummarizerNode invocation, linked to its trigger.

        ``triggered_by`` names the pipeline node that fired the summarizer
        (``finalize_step`` or ``context_guard``) so the viewer can nest the
        summarizer node under it in the trace tree. ``started_at`` is the
        wall-clock moment the summarization began, captured by the caller
        alongside its monotonic latency timer. ``resources`` is the caller's
        sampler-window summary for the same bracket; omitted when ``None``.
        """
        event: dict[str, Any] = {
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
        if resources is not None:
            event["resources"] = resources
        self._emit(event)

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

    def record_anomaly(
        self,
        kind: str,
        streak_id: str,
        signature: str,
        tool_name: str,
        streak_len: int,
        tokens_spent: int,
        step: int,
    ) -> None:
        """Record one anomaly detection row (passive: reports what detection found).

        ``streak_id`` is stable across the growth of one streak so later rows
        with the same id supersede earlier ones for consumers keeping the
        latest row per streak (the recorder itself does not dedupe or judge —
        it only appends what it is told).
        """
        self._emit(
            {
                "type": ANOMALY,
                "kind": kind,
                "streak_id": streak_id,
                "signature": signature,
                "tool_name": tool_name,
                "streak_len": streak_len,
                "tokens_spent": tokens_spent,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_transition(
        self,
        source: str,
        target: str,
        round_num: int,
        outcome: str,
        step: int,
    ) -> None:
        """Record one declared node-to-node transition (passive: reports the
        engine's already-made decision — see ``Pipeline._resolve_jump``).

        ``round_num`` is this backward transition's counter value at the time
        it was taken (0 for a forward transition, which is never bounded).
        ``outcome`` is ``"jump"`` (execution moved to ``target``) or
        ``"bound_exhausted"`` (the loop's ``max_rounds`` ceiling was reached
        and the run stopped instead). One row per taken transition and one for
        each bound exhaustion (issue #228), so counting rows per
        ``source``/``target`` pair reconstructs the full round history from
        ``events.jsonl`` alone.
        """
        self._emit(
            {
                "type": TRANSITION,
                "source": source,
                "target": target,
                "round": round_num,
                "outcome": outcome,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_refusal(
        self,
        tool_name: str,
        command: str,
        rule_id: str,
        reason: str,
        references: list[dict[str, str]],
        safer_alternative: str,
        step: int,
    ) -> None:
        """Record one policy-refusal row (passive: reports what the gate refused).

        Follows ``record_anomaly``'s template with a true wall-clock
        ``started_at``. The recorder never participates in the refusal
        decision — it only appends what the gate (``tool_execution.policy``)
        already decided, via the executor.
        """
        self._emit(
            {
                "type": REFUSAL,
                "tool_name": tool_name,
                "command": command,
                "rule_id": rule_id,
                "reason": reason,
                "references": references,
                "safer_alternative": safer_alternative,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_egress(
        self,
        tool_name: str,
        host: str,
        matched_list: str,
        reason: str,
        step: int,
    ) -> None:
        """Record one egress-denial row (passive: reports what the filter blocked).

        Follows ``record_refusal``'s template: the recorder never participates
        in the block decision — it only appends what ``engine.egress`` already
        decided, via the executor.
        """
        self._emit(
            {
                "type": EGRESS,
                "tool_name": tool_name,
                "host": host,
                "matched_list": matched_list,
                "reason": reason,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_hook(
        self,
        event: str,
        hook_name: str,
        outcome: str,
        step: int,
        tool_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record one lifecycle-hook firing (passive: reports what fired/blocked).

        Follows ``record_refusal``'s template with a true wall-clock
        ``started_at``. The recorder never participates in the hook's
        decision — it only appends what the hook mechanism (``engine.hooks``)
        already decided, via the executor / ``AgentNode``.
        """
        self._emit(
            {
                "type": HOOK,
                "event": event,
                "hook_name": hook_name,
                "tool_name": tool_name,
                "outcome": outcome,
                "reason": reason,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_sandbox_activation(
        self, workspace_root: str, extra_write_paths: list[str]
    ) -> None:
        """Record that this run's ``bash`` subprocesses are OS-sandboxed.

        Emitted once, at session start, only when the sandbox is enabled — a
        sandbox-off run emits no row, so its ``events.jsonl`` stays byte-
        identical to a pre-change trace. Passive: reports the scope the
        sandbox module already computed, never influences enforcement.
        """
        self._emit(
            {
                "type": SANDBOX_ACTIVATION,
                "workspace_root": workspace_root,
                "extra_write_paths": extra_write_paths,
                "started_at": _now(),
            }
        )

    def record_sandbox_denial(
        self, command: str, exit_code: int, stderr: str, step: int
    ) -> None:
        """Record one sandboxed ``bash`` call the OS denied.

        Follows ``record_refusal``'s template. The recorder never decides
        what counts as a denial — the caller (``registry.bash``) already
        classified the failed, sandboxed command before calling this.
        """
        self._emit(
            {
                "type": SANDBOX_DENIAL,
                "command": command,
                "exit_code": exit_code,
                "stderr": stderr,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_provenance(
        self,
        kind: str,
        tool_name: str,
        reason: str,
        step: int,
    ) -> None:
        """Record one provenance row: an ingestion-time untrusted mark, or a
        capability-reduction boundary refusal (passive; issue #128).

        Follows ``record_refusal``'s template. Never echoes the ingested
        content itself — only the tool name and a short reason are captured.
        """
        self._emit(
            {
                "type": PROVENANCE,
                "kind": kind,
                "tool_name": tool_name,
                "reason": reason,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_exfil(
        self,
        tool_name: str,
        category: str,
        step: int,
    ) -> None:
        """Record one exfiltration-guard block (passive: reports what the
        guard blocked). Follows ``record_refusal``'s template — the recorder
        never participates in the block decision, and never receives (so
        never can leak) the matched secret value, only its category name.
        """
        self._emit(
            {
                "type": EXFIL,
                "tool_name": tool_name,
                "category": category,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_supersession(
        self,
        tool_call_id: str,
        tool_name: str,
        case: str,
        superseding_tool_call_id: str,
        retired_size: int,
        step: int,
    ) -> None:
        """Record one tool-result retirement (passive: reports what retired).

        Emitted once per retired message by ``ContextGuardNode``'s
        supersession pass, never influencing execution — the recorder only
        appends what it is told, same as ``record_anomaly``.
        """
        self._emit(
            {
                "type": SUPERSESSION,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "case": case,
                "superseding_tool_call_id": superseding_tool_call_id,
                "retired_size": retired_size,
                "step": step,
                "started_at": _now(),
            }
        )

    def record_skill_index(self, names: list[str], chars: int, tier: str) -> None:
        """Record that a skill index was placed into an opening user message.

        Emitted once at session start and once per continuation when the index
        is placed (skill-knowledge-delivery D9) — the *offered* record. Purely
        passive: it captures what the run offered, and never influences
        execution. Absent entirely from a skill-free run's ``events.jsonl`` (the
        caller only records when skills exist), so pre-skill traces parse
        identically to today.

        Args:
            names: Skill names listed in the placed index.
            chars: Size in characters of the block actually placed (including any
                re-injected loaded-skill bodies on a continuation).
            tier: Degradation tier applied to fit the budget
                (``full`` / ``truncated`` / ``names_only``).
        """
        self._emit(
            {
                "type": SKILL_INDEX,
                "names": names,
                "count": len(names),
                "chars": chars,
                "tier": tier,
                "started_at": _now(),
            }
        )

    # ── LLM outage recovery (run-resilience D2) ─────────────────────────────────
    def record_llm_wait(
        self,
        kind: str,
        call: int,
        classification: str,
        attempt: int,
        delay_s: float,
        elapsed_s: float,
    ) -> None:
        """Record one patient-phase wait before an outage retry (passive)."""
        self._emit(
            {
                "type": LLM_WAIT,
                "kind": kind,
                "call": call,
                "classification": classification,
                "attempt": attempt,
                "delay_s": delay_s,
                "elapsed_s": elapsed_s,
                "started_at": _now(),
            }
        )

    def record_llm_recovery(
        self, kind: str, call: int, attempts: int, stalled_s: float
    ) -> None:
        """Record that the server answered after an absorbed outage (passive)."""
        self._emit(
            {
                "type": LLM_RECOVERY,
                "kind": kind,
                "call": call,
                "attempts": attempts,
                "stalled_s": stalled_s,
                "started_at": _now(),
            }
        )

    def record_llm_failure(
        self,
        kind: str,
        call: int,
        classification: str,
        attempts: int,
        elapsed_s: float,
    ) -> None:
        """Record an unrecoverable LLM failure that ends the run (passive)."""
        self._emit(
            {
                "type": LLM_FAILURE,
                "kind": kind,
                "call": call,
                "classification": classification,
                "attempts": attempts,
                "elapsed_s": elapsed_s,
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
        return {
            "content": "",
            "reasoning": "",
            "tool_calls": [],
            "raw": {},
            "finish_reason": None,
        }
    msg = choices[0].get("message", {}) or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "raw": msg,
        "finish_reason": choices[0].get("finish_reason"),
    }

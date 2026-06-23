"""Reconstruct the hierarchical pipeline trace tree from a session's events.

``build_trace_tree`` turns a parsed :class:`Session` into a nested
:class:`TreeNode` whose shape mirrors the runtime pipeline: each ``Agent`` run is
a root; its steps nest ``ContextManager`` → ``ToolRouter`` → the LLM call (whose
reasoning/content land in its ``output``) → one ``ToolExecutor`` per tool call
(with ``ToolOutputValidation`` children). Delegated subagents nest under the
``ToolExecutor`` of their ``delegate`` call via ``child_session_id``.

The whole pipeline is harness-managed, so nodes carry no "creator". Titles are
concise class/function names (facts live in metadata). Each node exposes
``status``/``message``/``input``/``output`` for the viewer's decision panel, and
``annotate_context`` adds a per-node ``ctx`` bar (history + tokens added/removed).
"""

from __future__ import annotations

import json
from typing import Any

from . import pricing
from .events import Session, ToolCall, TreeNode

_VALIDATION: dict[str, tuple[str, str, str]] = {
    # raw kind -> (function name, status, message)
    "tool_output_summarizer": (
        "output.summarize_artifact",
        "success",
        "output.summarize_artifact: summarized large output",
    ),
    "tool_arg_correction": (
        "args.correct_args",
        "warning",
        "args.correct_args: corrected tool arguments",
    ),
}

# Node types whose token "added" figure is a length estimate, not an exact count.
_ESTIMATED = {"tool_executor", "user_message", "system_message"}

# LLM kinds that run on their own ephemeral conversation (not the agent's window),
# so they must not be counted as tokens added to the agent's context.
_ANCILLARY_LLM_KINDS = {"tool_router", "tool_output_summarizer", "tool_arg_correction"}

# The only job-status values; every node is normalized to one of these so the
# viewer can show a single status logo (green check / red cross / amber warning).
_STATUS_SET = {"success", "failure", "warning"}


def _node(type_: str, title: str, *, timestamp: str = "", **metadata: Any) -> TreeNode:
    """Construct a ``TreeNode`` (metadata keys become the node's metadata)."""
    return TreeNode(type=type_, title=title, timestamp=timestamp, metadata=metadata)


def build_trace_tree(
    session: Session,
    by_id: dict[str, Session] | None = None,
    consumed: set[str] | None = None,
) -> TreeNode:
    """Build the trace tree for ``session`` (recursing into delegated children)."""
    top = consumed is None
    by_id = by_id or {}
    consumed = consumed if consumed is not None else set()
    consumed.add(session.session_id)

    cost = pricing.cost_usd(
        session.model, session.prompt_tokens, session.completion_tokens
    )
    root = _node(
        "agent",
        f"Agent: {session.label}",
        timestamp=session.started_at,
        status="success" if session.ok else "failure",
        message=f"run stopped: {session.stop_reason}",
        label=session.label,
        session_id=session.session_id,
        model=session.model,
        parent_session_id=session.parent_session_id,
        stop_reason=session.stop_reason,
        steps=session.steps,
        elapsed_s=session.elapsed_s,
        total_tokens=session.total_tokens,
        est_cost_usd=round(cost, 6),
        context_window=session.context_window,
    )

    for msg in _initial_messages(session):
        root.add(_message_node(msg))

    state = _StepBuilder(root, session, by_id, consumed)
    for ev in session.timeline:
        state.feed(ev)

    # Handoff continuations are fresh agents linked by parent_session_id (not a
    # delegate call); append any not already nested as delegated children.
    for child_id, child in by_id.items():
        if child.parent_session_id == session.session_id and child_id not in consumed:
            root.add(build_trace_tree(child, by_id, consumed))

    if top:
        _reindex(root, "0")
        annotate_context(root)
        _normalize_status(root)
    return root


def _normalize_status(node: TreeNode) -> None:
    """Give every node a status logo: default to ``success`` when none applies.

    Nodes like ``ToolRouter`` / ``ContextManager`` / messages carry no failure
    signal; they completed, so they read ``success``. Existing ``failure`` /
    ``warning`` values are preserved.
    """
    if node.metadata.get("status") not in _STATUS_SET:
        node.metadata["status"] = "success"
    for child in node.children:
        _normalize_status(child)


def _initial_messages(session: Session) -> list[dict[str, Any]]:
    """Leading system/user messages from the first snapshot-bearing call."""
    first = next((c for c in session.llm_calls if c.messages), None)
    if not first or not first.messages:
        return []
    out: list[dict[str, Any]] = []
    for m in first.messages:
        if m.get("role") in ("system", "user"):
            out.append(m)
        else:
            break
    return out


def _message_node(msg: dict[str, Any]) -> TreeNode:
    """Build a SystemMessage / UserMessage node."""
    content = msg.get("content") or ""
    if msg.get("role") == "system":
        return _node("system_message", "Agent.add_message (system)", output=content)
    return _node("user_message", "Agent.add_message (user)", output=content)


class _StepBuilder:
    """Stateful walk of the event timeline that assembles step subtrees."""

    def __init__(
        self,
        root: TreeNode,
        session: Session,
        by_id: dict[str, Session],
        consumed: set[str],
    ) -> None:
        self.root = root
        self.session = session
        self.by_id = by_id
        self.consumed = consumed
        self.step: TreeNode | None = None
        self.index = 0
        self.tool_idx = 0
        self._router_event: dict[str, Any] | None = None
        self._router_call: dict[str, Any] | None = None
        self._validation: list[dict[str, Any]] = []

    def feed(self, ev: dict[str, Any]) -> None:
        """Dispatch one timeline event to the matching handler."""
        etype = ev.get("type")
        if etype == "router":
            self._open_step(ev)
        elif etype == "llm_call":
            self._on_llm_call(ev)
        elif etype == "tool_call":
            self._on_tool_call(ev)
        elif etype == "handoff":
            (self.step or self.root).add(_handoff_node(ev))

    def _new_step(self, ev: dict[str, Any]) -> None:
        self.index += 1
        self.step = self.root.add(
            _node("step", f"Step {self.index}", timestamp=ev.get("started_at", ""))
        )

    def _open_step(self, ev: dict[str, Any]) -> None:
        self._new_step(ev)
        self._router_event = ev  # rendered (after ContextManager) on the main call

    def _on_llm_call(self, ev: dict[str, Any]) -> None:
        kind = ev.get("kind")
        if kind == "main":
            if self.step is None:
                self._new_step(ev)
            assert self.step is not None
            self.step.add(_context_node(ev))
            if self._router_event is not None:
                self.step.add(_router_node(self._router_event, self._router_call))
                self._router_event = None
                self._router_call = None
            self.step.add(_llm_node(ev))
        elif kind == "tool_router":
            self._router_call = ev
        elif kind in ("tool_output_summarizer", "tool_arg_correction"):
            self._validation.append(ev)

    def _on_tool_call(self, ev: dict[str, Any]) -> None:
        if self.step is None:
            self._new_step(ev)
        assert self.step is not None
        record = (
            self.session.tool_calls[self.tool_idx]
            if self.tool_idx < len(self.session.tool_calls)
            else None
        )
        self.tool_idx += 1
        # One ToolExecutor node per tool call (no separate ToolCall level).
        tnode = self.step.add(_tool_node(ev, record, self.session))
        for v in self._validation:
            tnode.add(_validation_node(v))
        self._validation = []
        _maybe_nest_child(tnode, ev, self.by_id, self.consumed)


def _context_node(ev: dict[str, Any]) -> TreeNode:
    """ContextManager node derived from a main call's prompt/window usage.

    Input is the full conversation being measured; output is the preflight
    decision (proceed vs. reset) with the exact token accounting.
    """
    prompt = ev.get("prompt", 0)
    window = ev.get("context_window", 0)
    pct = round(prompt / window * 100, 1) if window else 0.0
    decision = f"proceed — {prompt:,}/{window:,} tokens ({pct}%) below reset threshold"
    return _node(
        "context_manager",
        "Agent._context_preflight",
        timestamp=ev.get("started_at", ""),
        message=f"_context_preflight: {prompt:,}/{window:,} tokens ({pct}%)",
        input=ev.get("messages") or [],
        output=decision,
        prompt_tokens=prompt,
        context_window=window,
        context_pct=pct,
    )


def _handoff_node(ev: dict[str, Any]) -> TreeNode:
    """ContextManager node for a context-reset (window full → evicted)."""
    return _node(
        "context_manager",
        "Agent._handle_context_reset",
        timestamp=ev.get("started_at", ""),
        status="warning",
        message=f"_handle_context_reset: evicted at {ev.get('ctx_pct')}%",
        reset=True,
        ctx_tokens=ev.get("ctx_tokens"),
        ctx_pct=ev.get("ctx_pct"),
        output=ev.get("content", ""),
    )


def _router_node(router_ev: dict[str, Any], call_ev: dict[str, Any] | None) -> TreeNode:
    """ToolRouter node with the selected tool subset and routing phase."""
    used = router_ev.get("used_llm", False)
    phase = router_ev.get("phase")
    node = _node(
        "tool_router",
        "ToolRouter.route_tools",
        timestamp=router_ev.get("started_at", ""),
        status="info" if used else None,
        message=f"route_tools: {phase}" + (" (LLM fallback)" if used else ""),
        input=router_ev.get("signal", ""),
        output=router_ev.get("selected", []) or [],
        phase=phase,
        used_llm_fallback=used,
    )
    # When routing fell back to the LLM, nest that call so its input/output and
    # token cost are visible like any other LLM.chat_completion.
    if call_ev is not None:
        node.add(_llm_node(call_ev))
    return node


def _llm_node(ev: dict[str, Any]) -> TreeNode:
    """LLM-call node; its output is the raw assistant response from the server."""
    resp = ev.get("response", {}) or {}
    content = resp.get("content") or ""
    # The unmodified LLM response message (falls back to the summary if a session
    # predates raw capture).
    raw = resp.get("raw") or {
        "content": content,
        "reasoning_content": resp.get("reasoning") or "",
        "tool_calls": resp.get("tool_calls") or [],
    }
    empty = not content and not resp.get("tool_calls")
    node = _node(
        "llm_call",
        "LLM.chat_completion",
        timestamp=ev.get("started_at", ""),
        status="warning" if empty else "success",
        message="chat_completion" + (" — empty response" if empty else ""),
        kind=ev.get("kind"),
        call=ev.get("call"),
        input=ev.get("messages") or [],
        output=raw,
        prompt_tokens=ev.get("prompt", 0),
        completion_tokens=ev.get("completion", 0),
        total_tokens=ev.get("total", 0),
        latency_s=ev.get("latency_s", 0),
    )
    if resp.get("tool_calls"):
        node.metadata["requested_tool_calls"] = [
            tc.get("function", {}).get("name") for tc in resp["tool_calls"]
        ]
    return node


def _sig(name: str, args: dict[str, Any]) -> str:
    """Stable signature of a tool call for duplicate detection."""
    return f"{name}({json.dumps(args, sort_keys=True, default=str)})"


def _result_failed(result: Any) -> bool:
    """True when a structured tool result reports failure without raising.

    Tools like ``bash`` return their outcome as data (``{"ok": false,
    "exit_code": N}``) rather than raising, so a failed command is otherwise
    indistinguishable from a successful one. Detect that here so the node can show
    the error logo like any other failure.
    """
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("ok") is False:
        return True
    exit_code = data.get("exit_code")
    return isinstance(exit_code, int) and exit_code != 0


def _tool_status(
    record: ToolCall | None, name: str, result: Any = ""
) -> tuple[str, str]:
    """Map a tool record (and its result) to a (status, message) for the panel."""
    raw = record.status if record else None
    if raw == "skipped":
        return "warning", f"invoke_tool({name}) skipped"
    if (record and record.ok is False) or raw == "error" or _result_failed(result):
        return "failure", f"invoke_tool({name}) failed"
    return "success", f"invoke_tool({name})"


def _tool_node(
    ev: dict[str, Any], record: ToolCall | None, session: Session
) -> TreeNode:
    """ToolExecutor node (one per tool call): input/output, latency, status."""
    name = str(ev.get("name", ""))
    args = ev.get("args", {}) or {}
    target = _sig(name, args)
    count = sum(1 for t in session.tool_calls if _sig(t.name, t.args) == target)
    status, message = _tool_status(record, name, ev.get("result", ""))
    if count > 1:
        status = "warning" if status == "success" else status
        message += f" · duplicate ×{count}"
    meta: dict[str, Any] = {
        "status": status,
        "message": message,
        "name": name,
        "input": args,
        "output": ev.get("result", ""),
        "latency_s": ev.get("latency_s", 0),
        "tool_status": record.status if record else None,
        "tool_call_id": record.tool_call_id if record else None,
    }
    if ev.get("child_session_id"):
        meta["child_session_id"] = ev["child_session_id"]
    if count > 1:
        meta["duplicate_count"] = count
    return _node(
        "tool_executor",
        f"ToolExecutor.invoke_tool: {name}",
        timestamp=ev.get("started_at", ""),
        **meta,
    )


def _validation_node(ev: dict[str, Any]) -> TreeNode:
    """Node for an ancillary executor helper wrapping the LLM call it makes.

    Both ``output.summarize_artifact`` and ``args.correct_args`` are real
    ``LLM.chat_completion`` calls; the call is nested as its own ``llm_call`` node
    so its prompt/response and token cost are visible in the tree.
    """
    kind = str(ev.get("kind", ""))
    fn, status, message = _VALIDATION.get(kind, (kind, "info", kind))
    node = _node(
        "tool_output_validation",
        f"ToolExecutor.{fn}",
        timestamp=ev.get("started_at", ""),
        status=status,
        message=message,
        function=fn,
    )
    node.add(_llm_node(ev))
    return node


def _maybe_nest_child(
    tnode: TreeNode,
    ev: dict[str, Any],
    by_id: dict[str, Session],
    consumed: set[str],
) -> None:
    """Nest a delegated subagent's tree under its delegate tool call."""
    child_id = ev.get("child_session_id")
    if child_id and child_id in by_id and child_id not in consumed:
        tnode.add(build_trace_tree(by_id[child_id], by_id, consumed))


def _reindex(node: TreeNode, prefix: str) -> None:
    """Assign globally stable dotted ``node_id`` values across the whole tree."""
    node.node_id = prefix
    for i, child in enumerate(node.children):
        _reindex(child, f"{prefix}.{i}")


# ── context-window annotation ──────────────────────────────────────────────────
def _est_tokens(text: Any) -> int:
    """Rough token estimate (~4 chars/token) for text not metered by the API."""
    s = str(text or "")
    return max(1, len(s) // 4) if s else 0


def _ctx_delta(node: TreeNode) -> tuple[int, int]:
    """Tokens this node adds / removes from its agent's running context.

    Exact where the API meters it (LLM responses → ``completion_tokens``; a reset
    evicts ``ctx_tokens``); estimated from length for tool results and the seed
    system/user messages.
    """
    t, md = node.type, node.metadata
    if t == "llm_call":
        if md.get("kind") in _ANCILLARY_LLM_KINDS:
            return 0, 0  # side-call, not part of the agent's running context
        return md.get("completion_tokens", 0) or 0, 0
    if t == "tool_executor":
        return _est_tokens(md.get("output", "")), 0
    if t in ("user_message", "system_message"):
        return _est_tokens(md.get("output", "")), 0
    if t == "context_manager" and md.get("reset"):
        return 0, md.get("ctx_tokens", 0) or 0
    return 0, 0


def annotate_context(root: TreeNode) -> None:
    """Attach a per-node ``ctx`` bar (history + tokens added/removed) to the tree.

    Walks each agent's subtree in order, re-anchoring the running context to each
    step's main-call ``prompt_tokens`` (the exact, API-metered value) so length
    estimates between calls self-correct. Each ``ctx`` also records the owning
    agent's name/id for the viewer's "CONTEXT WINDOW OF AGENT …" header.
    """
    _annotate(root, _CtxScope(0, "", ""), [0])


class _CtxScope:
    """The owning agent's context window + identity for a subtree."""

    def __init__(self, window: int, label: str, agent_id: str) -> None:
        self.window = window
        self.label = label
        self.agent_id = agent_id


def _annotate(node: TreeNode, scope: _CtxScope, acc: list[int]) -> None:
    if node.type == "agent":
        scope = _CtxScope(
            node.metadata.get("context_window", scope.window) or scope.window,
            node.metadata.get("label", ""),
            node.metadata.get("session_id", ""),
        )
        acc = [0]  # fresh running context for this agent's own window
    if node.type == "step":
        main = next((c for c in node.children if c.type == "llm_call"), None)
        if main is not None:
            acc[0] = main.metadata.get("prompt_tokens", acc[0]) or acc[0]
    added, removed = _ctx_delta(node)
    node.metadata["ctx"] = {
        "history": acc[0],
        "window": scope.window,
        "added": added,
        "removed": removed,
        "estimated": added > 0 and node.type in _ESTIMATED,
        "agent_label": scope.label,
        "agent_id": scope.agent_id,
    }
    if removed:
        acc[0] = max(0, acc[0] - removed)
    else:
        acc[0] += added
    for child in node.children:
        _annotate(child, scope, acc)

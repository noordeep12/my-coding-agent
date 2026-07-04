"""Parse ``events.jsonl`` into a ``TraceSession`` for the Trace Explorer.

Public API
----------
- ``load_session(events_path, _seen)`` — parse one session from disk.
- ``list_sessions(base_dir)``          — scan a directory for all sessions.

Internal helpers follow the single-responsibility pattern: each private
function has one job and can be tested in isolation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pricing import compute_cost
from .schema import EventBuilder, TraceNode, TraceSession

logger = logging.getLogger(__name__)


# ── Graph accumulator ─────────────────────────────────────────────────────────


@dataclass
class _Graph:
    """Mutable accumulator for the node graph built while parsing a session.

    Bundles ``nodes`` and the execution ``order`` so callers wire a new node
    with a single ``add`` call instead of threading both through every builder.
    Each added node is stamped with its owning ``agent`` and call-tree ``depth``
    so the UI can nest and badge sub-agent traces.

    Args:
        root_id: ID of the session root node; every added node parents to it.
        agent: Owning session id stamped onto every node.
        base_depth: Depth of this session's root; pipeline nodes sit one deeper.
        call_index: LLM call ordinal → node id, so a tool_call event's
            ``child_llm_calls`` (LLM calls made inside that tool's own
            implementation, e.g. read_tool_artifact's extraction call) can be
            re-parented under that exact tool node once both are known.
    """

    root_id: str
    agent: str
    base_depth: int = 0
    nodes: dict[str, TraceNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    call_index: dict[int, str] = field(default_factory=dict)

    def add_root(self, node: TraceNode) -> None:
        """Register the root node and start the execution order.

        Args:
            node: The session root ``TraceNode``.
        """
        node.agent = self.agent
        node.depth = self.base_depth
        self.nodes[node.id] = node
        self.order.append(node.id)

    def add(self, node_id: str, node: TraceNode, step: int | None = None) -> None:
        """Append *node* to the chain in execution order.

        Args:
            node_id: Stable node identifier.
            node: The ``TraceNode`` to add; its ``parent_id`` is set to the root.
            step: Optional 1-based step number stamped onto ``attributes``.
        """
        if step is not None:
            node.attributes["step"] = step
        node.parent_id = self.root_id
        node.agent = self.agent
        node.depth = self.base_depth + 1
        self.nodes[node_id] = node
        self.order.append(node_id)


# ── Public API ───────────────────────────────────────────────────────────────


def list_sessions(base_dir: Path) -> list[dict[str, Any]]:
    """Scan *base_dir* for session subdirectories and return summary rows.

    Each row contains ``session_id``, ``label``, ``model``, ``started_at``,
    ``stop_reason``, and ``steps``.  Sessions are returned newest-first by
    ``started_at``.  Sessions that have neither ``events.jsonl`` nor
    ``session_data.json`` are silently skipped.

    Args:
        base_dir: Root directory (typically ``.my_coding_agent/``).

    Returns:
        List of summary dicts, sorted descending by ``started_at``.

    Raises:
        OSError: If *base_dir* exists but cannot be listed (e.g. permission denied).
    """
    rows: list[dict[str, Any]] = []
    if not base_dir.is_dir():
        return rows
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        row = _summarise_dir(child)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return rows


def load_session(
    events_path: Path,
    _seen: set[str] | None = None,
    _depth: int = 0,
) -> TraceSession:
    """Parse *events_path* (``events.jsonl``) into a ``TraceSession``.

    When a ``delegate`` tool call references a child session, this function
    recurses to load and embed the child's trace.  The *_seen* guard prevents
    infinite recursion on cyclic session references.

    Falls back to ``session_data.json`` in the same directory when
    ``events.jsonl`` is absent, producing a minimal two-node trace.

    Args:
        events_path: Path to the ``events.jsonl`` file.
        _seen: Set of already-visited session IDs (internal recursion guard).
        _depth: Call-tree depth of this session's root (0 for the top-level
            agent; incremented when recursing into a delegate sub-agent).

    Returns:
        Fully parsed ``TraceSession`` with nodes in execution order.

    Raises:
        OSError: If *events_path* exists but cannot be read.  Callers should
            handle this when the file may be unreadable (e.g. permission errors
            in a multi-user session directory).
    """
    seen = _seen or set()
    session_dir = events_path.parent

    if not events_path.exists():
        return _fallback_session(session_dir, _depth)

    events = _read_events(events_path)
    start_ev = _find_start(events)
    session_id = start_ev.get("session_id", session_dir.name)

    if session_id in seen:
        return _stub_session(session_id, _depth)
    seen.add(session_id)

    end_ev = _find_end(events)
    steps_groups = _group_into_steps(events)

    root_id = f"{session_id}::session"
    graph = _Graph(root_id=root_id, agent=session_id, base_depth=_depth)
    graph.add_root(
        _make_node(
            id=root_id,
            type="session",
            label=start_ev.get("label", "Session"),
            inputs={},
            outputs={},
            attributes={
                "model": start_ev.get("model", ""),
                "context_window": start_ev.get("context_window"),
                "started_at": start_ev.get("started_at", ""),
                "parent_session_id": start_ev.get("parent_session_id"),
            },
        )
    )

    # Flat chain: every pipeline node is a direct child of the session root,
    # appended in execution order.  The step number is kept as an attribute
    # (for the Tree view) rather than a wrapper node, so the trace reads as one
    # continuous flow of RunContext through the pipeline.
    for step_idx, group in enumerate(steps_groups):
        _build_step_nodes(group, graph, session_id, step_idx, seen, session_dir)

    _add_anomaly_nodes(events, graph, session_id)
    _flag_anomalies(graph.nodes, graph.order, events)

    if end_ev:
        end_id = f"{session_id}::session_end"
        graph.add(
            end_id,
            _make_node(
                id=end_id,
                type="session_end",
                label="End",
                inputs={},
                outputs={},
                attributes={
                    "stop_reason": end_ev.get("stop_reason", ""),
                    "steps": end_ev.get("steps", 0),
                    "elapsed_s": end_ev.get("elapsed_s", 0.0),
                },
            ),
        )

    model = start_ev.get("model", "")
    _detect_loops(graph.nodes)
    # Only this session's own nodes; embedded sub-agent nodes already carry the
    # ctx_state computed by their own recursive load, with their own window.
    _assign_ctx_state(
        graph.nodes, graph.order, start_ev.get("context_window"), session_id
    )
    analytics = _compute_analytics(graph.nodes, model, end_ev)

    return TraceSession(
        session_id=session_id,
        label=start_ev.get("label", "Session"),
        model=model,
        started_at=start_ev.get("started_at", ""),
        ended_at=end_ev.get("ended_at") if end_ev else None,
        stop_reason=end_ev.get("stop_reason") if end_ev else None,
        steps=len(steps_groups),
        nodes=graph.nodes,
        order=graph.order,
        analytics=analytics,
    )


# ── Step grouping ─────────────────────────────────────────────────────────────


def _group_into_steps(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split *events* into step buckets, one per ``router`` event.

    Events that appear before the first ``router`` are discarded (they belong
    to session-level setup already captured in ``session_start``).

    Args:
        events: Raw list of JSONL event dicts.

    Returns:
        List of groups; each group starts with a ``router`` event.
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("type") == "router":
            if current:
                groups.append(current)
            current = [ev]
        elif current:
            current.append(ev)
    if current:
        groups.append(current)
    return groups


# ── Node building ─────────────────────────────────────────────────────────────


def _merge_finalize_events(
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse a step's ``token_tracking`` + ``finish_check`` events into one.

    ``FinalizeStepNode`` records token usage and the finish check as two separate
    events; the trace should show them as a single node.  Replace both (whichever
    are present) with one ``finalize_step`` event, merging their attributes, kept
    at the position of the first — preserving execution order.
    """
    finalize = {
        k: v
        for ev in group
        if ev.get("type") in ("token_tracking", "finish_check")
        for k, v in ev.items()
        if k != "type"
    }
    if not finalize:
        return group

    out: list[dict[str, Any]] = []
    inserted = False
    for ev in group:
        if ev.get("type") in ("token_tracking", "finish_check"):
            if not inserted:
                out.append({"type": "finalize_step", **finalize})
                inserted = True
            continue
        out.append(ev)
    return out


def _build_step_nodes(
    group: list[dict[str, Any]],
    graph: _Graph,
    session_id: str,
    step_idx: int,
    seen: set[str],
    session_dir: Path,
) -> None:
    """Append one pipeline step's nodes to the flat session chain in *graph*.

    Each node becomes a direct child of the session root and links to the node
    that ran immediately before it.  The 1-based step number is stamped onto
    every node's ``attributes["step"]`` so the Tree view can group them without
    a wrapper node.

    Args:
        group: All events belonging to this step (starts with ``router``).
        graph: Mutable graph accumulator the nodes are added to.
        session_id: Owning session ID (used for stable node IDs).
        step_idx: Zero-based step index.
        seen: Recursion guard set for delegate sessions.
        session_dir: Filesystem directory of the owning session.
    """
    step = step_idx + 1
    counters = {"llm": 0, "tool": 0}

    for ev in _merge_finalize_events(group):
        built = _build_event_node(ev, session_id, step, counters)
        if built is None:
            continue
        node_id, node = built
        graph.add(node_id, node, step=step)

        if node.type == "llm_call":
            call_num = ev.get("call")
            if call_num is not None:
                graph.call_index[call_num] = node_id

        if node.type == "tool_call":
            _reparent_child_llm_calls(node_id, node, ev.get("child_llm_calls"), graph)

            child_sid = node.attributes.get("child_session_id")
            if child_sid and child_sid not in seen:
                _embed_child_session(child_sid, graph, seen, session_dir, node_id)

    _nest_summarizer_nodes(graph, session_id, step)


def _reparent_child_llm_calls(
    tool_node_id: str,
    tool_node: TraceNode,
    call_numbers: list[int] | None,
    graph: _Graph,
) -> None:
    """Nest LLM calls a tool made internally under that tool's own node.

    ``call_numbers`` are the ordinals the recorder stashed while the tool was
    dispatching (e.g. read_tool_artifact's bounded ``artifact_query``
    extraction call). Each was already added as a flat, root-parented
    ``llm_call`` node earlier in this same event stream (LLM calls happen
    inside the tool's implementation, before its own ``tool_call`` event is
    emitted) — this re-parents it under the tool node instead, one level
    deeper, so the Tree view nests it the same way a delegate's subagent
    session nests under its ``delegate`` tool call.
    """
    for call_num in call_numbers or []:
        child_id = graph.call_index.get(call_num)
        child_node = graph.nodes.get(child_id) if child_id else None
        if child_node is None:
            continue
        child_node.parent_id = tool_node_id
        child_node.depth = tool_node.depth + 1


def _nest_summarizer_nodes(graph: _Graph, session_id: str, step: int) -> None:
    """Nest a step's summarizer under its trigger, and its LLM call under it.

    When a ``summarizer`` event was recorded in this step, re-parent the
    ContextSummarizerNode under the node that triggered it (``finalize_step``
    for a step-ceiling cutoff, the ``handoff`` node — labelled ContextGuardNode
    — for a context reset), move it right after that trigger in execution
    order, and tuck the summarization's own ``report``/``handoff``-kind LLM
    call beneath it. Flat is the default: sessions without summarizer events
    (all legacy traces) are untouched, as is a summarizer whose trigger has no
    node in the tree (e.g. a context-limit stop, which records no handoff).
    """
    summarizer_id = f"{session_id}::step{step}::summarizer"
    summarizer = graph.nodes.get(summarizer_id)
    if summarizer is None:
        return
    triggered_by = summarizer.attributes.get("triggered_by", "")
    trigger_suffix = "finalize_step" if triggered_by == "finalize_step" else "handoff"
    trigger_id = f"{session_id}::step{step}::{trigger_suffix}"
    trigger = graph.nodes.get(trigger_id)
    if trigger is None:
        return
    summarizer.parent_id = trigger_id
    summarizer.depth = trigger.depth + 1
    moved = [summarizer_id]
    kind = summarizer.attributes.get("kind")
    llm_prefix = f"{session_id}::step{step}::llm::"
    llm_id = next(
        (
            nid
            for nid in graph.order
            if nid.startswith(llm_prefix)
            and graph.nodes[nid].attributes.get("kind") == kind
        ),
        None,
    )
    if llm_id is not None:
        graph.nodes[llm_id].parent_id = summarizer_id
        graph.nodes[llm_id].depth = summarizer.depth + 1
        moved.append(llm_id)
    for nid in moved:
        graph.order.remove(nid)
    insert_at = graph.order.index(trigger_id) + 1
    graph.order[insert_at:insert_at] = moved


# ── Per-event-type node builders (dispatched via _EVENT_BUILDERS) ─────────────


def _build_router_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the ToolRoutingNode node for a ``router`` event."""
    node_id = f"{session_id}::step{step}::router"
    return node_id, _make_node(
        id=node_id,
        type="router",
        label="ToolRoutingNode",
        inputs={"signal": ev.get("signal", "")[:120]},
        outputs={"selected": ev.get("selected", [])},
        attributes={
            "phase": ev.get("phase", ""),
            "used_llm": ev.get("used_llm", False),
            "started_at": ev.get("started_at", ""),
        },
    )


def _build_llm_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the LLMCallNode node for an ``llm_call`` event."""
    counters["llm"] += 1
    kind = ev.get("kind", "main")
    node_id = f"{session_id}::step{step}::llm::{counters['llm']}"
    label = "LLMCallNode" + (f" ({kind})" if kind != "main" else "")
    resp = ev.get("response") or {}
    max_tokens = ev.get("max_tokens")
    completion_tokens = ev.get("completion")
    finish_reason = resp.get("finish_reason")
    capped = max_tokens is not None and (
        finish_reason == "length"
        or (completion_tokens is not None and completion_tokens >= max_tokens)
    )
    return node_id, _make_node(
        id=node_id,
        type="llm_call",
        label=label,
        inputs={
            "messages": ev.get("messages") or [],
            "tools": ev.get("tools") or [],
        },
        outputs={
            "content": resp.get("content", ""),
            "reasoning": resp.get("reasoning", ""),
            "tool_calls": resp.get("tool_calls", []),
        },
        attributes={
            "call": ev.get("call"),
            "kind": kind,
            "latency_s": ev.get("latency_s"),
            "prompt_tokens": ev.get("prompt"),
            "completion_tokens": completion_tokens,
            "total_tokens": ev.get("total"),
            "context_window": ev.get("context_window"),
            "started_at": ev.get("started_at", ""),
            "max_tokens": max_tokens,
            "capped": capped,
        },
    )


def _build_tool_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the ToolDispatchNode node for a ``tool_call`` event."""
    counters["tool"] += 1
    node_id = f"{session_id}::step{step}::tool::{counters['tool']}"
    tool_name = ev.get("name", "tool")
    return node_id, _make_node(
        id=node_id,
        type="tool_call",
        label=f"ToolDispatchNode ({tool_name})",
        inputs={"args": ev.get("args", {})},
        outputs={"result": ev.get("result", "")},
        attributes={
            "name": tool_name,
            "latency_s": ev.get("latency_s"),
            "started_at": ev.get("started_at", ""),
            "child_session_id": ev.get("child_session_id"),
        },
    )


def _build_handoff_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the ContextGuardNode node for a ``handoff`` event."""
    node_id = f"{session_id}::step{step}::handoff"
    return node_id, _make_node(
        id=node_id,
        type="handoff",
        label="ContextGuardNode",
        inputs={},
        outputs={"content": ev.get("content", "")},
        attributes={
            "ctx_tokens": ev.get("ctx_tokens"),
            "ctx_pct": ev.get("ctx_pct"),
            "path": ev.get("path", ""),
            "started_at": ev.get("started_at", ""),
        },
    )


def _build_report_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the subagent report node for a ``report`` event.

    ``source`` surfaces the report's cost provenance (verbatim / summarizer /
    fallback); a pre-provenance event has no ``source`` key, which becomes an
    explicit ``"unknown"`` here rather than an error or a guessed path (D3).
    """
    node_id = f"{session_id}::step{step}::report"
    return node_id, _make_node(
        id=node_id,
        type="report",
        label="Subagent Report",
        inputs={},
        outputs={"content": ev.get("content", "")},
        attributes={
            "source": ev.get("source", "unknown"),
            "started_at": ev.get("started_at", ""),
        },
    )


def _build_summarizer_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the ContextSummarizerNode node for a ``summarizer`` event."""
    node_id = f"{session_id}::step{step}::summarizer"
    return node_id, _make_node(
        id=node_id,
        type="summarizer",
        label="ContextSummarizerNode",
        inputs={
            "kind": ev.get("kind", ""),
            "triggered_by": ev.get("triggered_by", ""),
        },
        outputs={},
        attributes={
            "kind": ev.get("kind", ""),
            "triggered_by": ev.get("triggered_by", ""),
            "latency_s": ev.get("latency_s"),
            "prompt_tokens": ev.get("prompt_tokens"),
            "completion_tokens": ev.get("completion_tokens"),
            "total_tokens": ev.get("total_tokens"),
            "started_at": ev.get("started_at", ""),
        },
    )


def _build_finalize_step_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the single FinalizeStepNode node for a merged ``finalize_step`` event.

    ``FinalizeStepNode`` emits two records per step — ``token_tracking`` then
    ``finish_check`` — which ``_merge_finalize_events`` combines into one
    ``finalize_step`` event carrying both attribute sets, so the trace shows a
    single node with token usage and the finish signal together.
    """
    node_id = f"{session_id}::step{step}::finalize_step"
    return node_id, _make_node(
        id=node_id,
        type="finalize_step",
        label="FinalizeStepNode",
        inputs={},
        outputs={},
        attributes={
            "prompt_tokens": ev.get("prompt_tokens"),
            "completion_tokens": ev.get("completion_tokens"),
            "total_tokens": ev.get("total_tokens"),
            "ctx_pct": ev.get("ctx_pct"),
            "context_window": ev.get("context_window"),
            "finish_reason": ev.get("finish_reason"),
            "signal": ev.get("signal"),
            "started_at": ev.get("started_at", ""),
        },
    )


# Lookup table: event type → builder (CONTRIBUTE.md §38 — table over if-chain).
_EVENT_BUILDERS: dict[str, EventBuilder] = {
    "router": _build_router_node,
    "llm_call": _build_llm_node,
    "tool_call": _build_tool_node,
    "handoff": _build_handoff_node,
    "report": _build_report_node,
    "summarizer": _build_summarizer_node,
    "finalize_step": _build_finalize_step_node,
}


def _build_event_node(
    ev: dict[str, Any],
    session_id: str,
    step: int,
    counters: dict[str, int],
) -> tuple[str, TraceNode] | None:
    """Build the ``TraceNode`` for one pipeline event, or ``None`` to skip it.

    Dispatches through ``_EVENT_BUILDERS``; *counters* is mutated by the LLM and
    tool builders to keep per-step ordinals stable.

    Args:
        ev: One raw event dict from ``events.jsonl``.
        session_id: Owning session ID (used for stable node IDs).
        step: 1-based step number.
        counters: Mutable ``{"llm": int, "tool": int}`` ordinal counters.

    Returns:
        ``(node_id, node)`` for a recognised event type, else ``None``.
    """
    builder = _EVENT_BUILDERS.get(ev.get("type", ""))
    return builder(ev, session_id, step, counters) if builder else None


def _embed_child_session(
    child_sid: str,
    graph: _Graph,
    seen: set[str],
    session_dir: Path,
    parent_tool_id: str,
) -> None:
    """Load a delegate child session and graft its nodes into *graph*.

    The child's nodes are appended to the parent's execution order right after
    the delegating ``tool_call`` node, so the sub-agent's trace appears inline.
    The child's root is explicitly re-parented to that ``tool_call`` node
    (``parent_tool_id``) so the Tree view nests the whole sub-agent under the
    exact ``delegate`` call that spawned it — the same mechanism
    ``_reparent_child_llm_calls`` uses to nest a tool's internal LLM calls
    under its own node. They arrive one call-tree level deeper, carrying their
    own ``agent`` id and their own per-agent ``ctx_state`` (computed by the
    recursive load).

    Args:
        child_sid: Session ID of the spawned sub-agent.
        graph: Mutable graph accumulator to extend.
        seen: Recursion guard.
        session_dir: Parent session directory (child is a sibling directory).
        parent_tool_id: Node id of the delegating ``tool_call`` (``delegate``).
    """
    child_dir = session_dir.parent / child_sid
    child_events = child_dir / "events.jsonl"
    child_session = load_session(child_events, _seen=seen, _depth=graph.base_depth + 1)
    graph.nodes.update(child_session.nodes)
    graph.order.extend(child_session.order)
    if child_session.order:
        child_root = graph.nodes.get(child_session.order[0])
        if child_root is not None:
            child_root.parent_id = parent_tool_id


# ── Context-window composition ────────────────────────────────────────────────

# The four message roles the context window is composed of, in display order.
_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")
# Tokens-per-character fallback (~4 chars/token) when no LLM call lets us measure.
_DEFAULT_TPC = 0.25


def _content_len(content: Any) -> int:
    """Return a character length for a message ``content`` of any shape.

    Args:
        content: A message's ``content`` — a string, ``None``, or a structured
            value (list/dict of content parts).

    Returns:
        Character count used as the attribution weight for this message.
    """
    if isinstance(content, str):
        return len(content)
    if content is None:
        return 0
    return len(json.dumps(content, default=str))


def _role_split(
    messages: list[dict[str, Any]], total_tokens: int | None
) -> dict[str, int] | None:
    """Split *total_tokens* across the four message roles by character share.

    Providers report one flat prompt-token total, so we attribute it to the
    ``system``/``user``/``assistant``/``tool`` roles by each role's share of the
    input characters.  Approximate, but always sums to the reported total.

    Args:
        messages: The call's input messages (each a ``{"role", "content"}`` dict).
        total_tokens: Provider-reported token total to distribute.

    Returns:
        ``{role: tokens}`` for non-empty roles, or ``None`` when there is nothing
        to attribute (no messages or no token total).
    """
    if not messages or not total_tokens:
        return None
    chars = {r: 0 for r in _ROLES}
    for msg in messages:
        role = msg.get("role", "")
        chars[role if role in chars else "user"] += _content_len(msg.get("content"))
    total = sum(chars.values())
    if total == 0:
        return None
    out = {r: round(total_tokens * chars[r] / total) for r in _ROLES if chars[r]}
    # Absorb rounding drift into the largest role so the parts sum exactly.
    drift = total_tokens - sum(out.values())
    if drift and out:
        out[max(out, key=lambda r: out[r])] += drift
    return out


def _tokens_per_char(nodes: list[TraceNode]) -> float:
    """Estimate a tokens-per-character ratio from a session's LLM calls.

    Used to size token figures for messages the provider never counted on their
    own — chiefly tool results.

    Args:
        nodes: One agent's trace nodes (in any order).

    Returns:
        Ratio of measured prompt tokens to input characters across main LLM
        calls, or ``_DEFAULT_TPC`` when nothing is measurable.
    """
    chars = toks = 0
    for node in nodes:
        if node.type != "llm_call" or node.attributes.get("kind") != "main":
            continue
        prompt = node.attributes.get("prompt_tokens")
        msgs = node.inputs.get("messages") or []
        if prompt and msgs:
            c = sum(_content_len(m.get("content")) for m in msgs)
            if c:
                chars += c
                toks += prompt
    return toks / chars if chars else _DEFAULT_TPC


def _first_input_split(nodes: list[TraceNode]) -> dict[str, int] | None:
    """Return the role split of the first main LLM call's input window.

    That window — system prompt plus the opening user message — is what the
    session node seeds the context with before any node runs.

    Args:
        nodes: One agent's trace nodes in execution order.

    Returns:
        ``{role: tokens}`` for the first call's input, or ``None``.
    """
    for node in nodes:
        if node.type == "llm_call" and node.attributes.get("kind") == "main":
            return _role_split(
                node.inputs.get("messages") or [],
                node.attributes.get("prompt_tokens"),
            )
    return None


def _node_added(
    node: TraceNode, comp: dict[str, int], first_snap: dict[str, int] | None, tpc: float
) -> tuple[dict[str, int], int, bool]:
    """Return what *node* appends to the context window.

    Each pipeline node may push one kind of message onto the running window: the
    session seeds it with the system prompt and opening user message, an LLM call
    appends its assistant reply, a tool dispatch appends its result.  Nodes that
    add nothing (router, finalize_step, …) return an empty delta.

    Args:
        node: The node being placed.
        comp: Running cumulative composition (mutated by the caller, read here).
        first_snap: Role split of the first LLM call's input (session seed).
        tpc: Tokens-per-character ratio for estimating uncounted text.

    Returns:
        ``(added, removed, estimated)`` — the per-role tokens this node added,
        tokens it compacted away, and whether any added figure is an estimate.
    """
    if node.type == "session":
        if first_snap:
            return (
                {r: first_snap[r] for r in ("system", "user") if first_snap.get(r)},
                0,
                False,
            )
        return {}, 0, False
    if node.type == "llm_call" and node.attributes.get("kind") == "main":
        snap = _role_split(
            node.inputs.get("messages") or [], node.attributes.get("prompt_tokens")
        )
        if snap:
            # Re-anchor the input roles to the real snapshot; the call's own new
            # output is layered on top below.
            for r in ("system", "user", "tool"):
                comp[r] = snap.get(r, 0)
            comp["assistant"] = snap.get("assistant", 0)
        completion = node.attributes.get("completion_tokens") or 0
        return ({"assistant": completion} if completion else {}), 0, False
    if node.type == "tool_call":
        tok = round(_content_len(node.outputs.get("result")) * tpc)
        return ({"tool": tok} if tok else {}), 0, bool(tok)
    if node.type == "handoff":
        ctx_tokens = node.attributes.get("ctx_tokens")
        old_total = sum(comp.values())
        if ctx_tokens is not None and old_total > ctx_tokens:
            sys_t = comp["system"]
            comp.update(
                system=sys_t,
                user=max(0, ctx_tokens - sys_t),
                assistant=0,
                tool=0,
            )
            return {}, old_total - ctx_tokens, False
    return {}, 0, False


def _assign_ctx_state(
    nodes: dict[str, TraceNode],
    order: list[str],
    default_window: int | None,
    owner: str,
) -> None:
    """Compute *owner*'s context-window composition, in execution order.

    Walks only the nodes belonging to *owner* (sub-agent nodes already carry the
    ctx_state from their own recursive load), maintaining the running window as
    four role buckets (system/user/assistant/tool).  Each node contributes the
    message(s) it pushes onto the window (``_node_added``); the cumulative
    composition and the per-node delta are stored on ``node.ctx_state``.

    Args:
        nodes: All trace nodes; *owner*'s nodes gain a ``ctx_state`` in-place.
        order: Node IDs in execution order (may include sub-agent nodes).
        default_window: Session context window size used when a node omits it.
        owner: Session id whose nodes this call should process.
    """
    own = [nodes[nid] for nid in order if nodes.get(nid) and nodes[nid].agent == owner]
    tpc = _tokens_per_char(own)
    first_snap = _first_input_split(own)
    comp = {r: 0 for r in _ROLES}
    window = default_window

    for node in own:
        own_window = node.attributes.get("context_window")
        if own_window:
            window = own_window

        added, removed, estimated = _node_added(node, comp, first_snap, tpc)
        for role, tok in added.items():
            comp[role] = comp.get(role, 0) + tok

        tokens = sum(comp.values())
        pct = round(tokens / window * 100, 1) if window else None
        node.ctx_state = {
            "tokens": tokens,
            "window": window,
            "pct": pct,
            "composition": {r: comp[r] for r in _ROLES if comp[r]},
            "added": added,
            "added_total": sum(added.values()),
            "removed": removed,
            "estimated": estimated,
        }


# ── Loop detection ────────────────────────────────────────────────────────────


def _detect_loops(nodes: dict[str, TraceNode]) -> None:
    """Set ``loop_flag=True`` on ``tool_call`` nodes whose (name, args) repeats.

    A signature is ``name + canonical JSON of args``.  Any tool_call that
    shares a signature with an earlier one is flagged — including the first
    occurrence once a repeat is found.

    Args:
        nodes: All nodes; modified in-place.
    """
    seen_sigs: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        if node.type != "tool_call":
            continue
        name = node.attributes.get("name", "")
        args = node.inputs.get("args", {})
        try:
            sig = name + "|" + json.dumps(args, sort_keys=True)
        except (TypeError, ValueError):
            sig = name
        seen_sigs.setdefault(sig, []).append(node_id)

    for ids in seen_sigs.values():
        if len(ids) > 1:
            for nid in ids:
                nodes[nid].loop_flag = True


# ── Anomaly rendering ──────────────────────────────────────────────────────────


def _latest_anomaly_by_streak(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the latest ``anomaly`` row per ``streak_id``, in first-seen order.

    A streak emits one row per growth (D3); consumers keep only the last row
    per ``streak_id`` since it carries the final, accumulated magnitude.
    """
    latest: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.get("type") == "anomaly":
            latest[ev.get("streak_id", "")] = ev
    return latest


def _add_anomaly_nodes(
    events: list[dict[str, Any]], graph: _Graph, session_id: str
) -> None:
    """Add one trace node per detected streak, keeping only its latest row."""
    for streak_id, ev in _latest_anomaly_by_streak(events).items():
        node_id = f"{session_id}::anomaly::{streak_id}"
        graph.add(
            node_id,
            _make_node(
                id=node_id,
                type="anomaly",
                label="AnomalyDetectNode",
                inputs={},
                outputs={},
                attributes={
                    "kind": ev.get("kind", ""),
                    "streak_id": streak_id,
                    "signature": ev.get("signature", ""),
                    "tool_name": ev.get("tool_name", ""),
                    "streak_len": ev.get("streak_len", 0),
                    "tokens_spent": ev.get("tokens_spent", 0),
                    "started_at": ev.get("started_at", ""),
                },
            ),
            step=ev.get("step"),
        )


def _is_failed_tool_result(result: str) -> bool:
    """Return ``True`` when a tool_call event's raw JSON result reports failure."""
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("ok") is False


def _flag_anomalies(
    nodes: dict[str, TraceNode], order: list[str], events: list[dict[str, Any]]
) -> None:
    """Set ``anomaly_flag=True`` on each streak's failing ``tool_call`` nodes.

    For each detected streak, walks the node order backwards from the end,
    matching consecutive ``tool_call`` nodes by tool name and failed result
    (D4's positional back-walk — ``tool_call`` events carry no ``tool_call_id``
    to link exactly). Stops once ``streak_len`` nodes are flagged or a
    non-matching ``tool_call`` node breaks the run.
    """
    for ev in _latest_anomaly_by_streak(events).values():
        tool_name = ev.get("tool_name", "")
        remaining = ev.get("streak_len", 0)
        for node_id in reversed(order):
            if remaining <= 0:
                break
            node = nodes[node_id]
            if node.type != "tool_call":
                continue
            if node.attributes.get("name") != tool_name or not _is_failed_tool_result(
                node.outputs.get("result", "")
            ):
                break
            node.anomaly_flag = True
            remaining -= 1


# ── Analytics ─────────────────────────────────────────────────────────────────


def _compute_analytics(
    nodes: dict[str, TraceNode],
    model: str,
    end_ev: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate token totals, cost, and event counts across all nodes.

    Args:
        nodes: All trace nodes.
        model: Model name used for cost estimation.
        end_ev: Raw ``session_end`` event dict, or ``None``.

    Returns:
        Dict with ``total_tokens``, ``total_prompt_tokens``,
        ``total_completion_tokens``, ``cost_usd``, ``elapsed_s``,
        ``llm_call_count``, ``tool_call_count``, ``loop_count``,
        ``anomaly_count``, ``by_kind`` (tokens per call kind) and ``by_agent``
        (per session id: tokens, wall time, call count).

    Token totals cover every call kind (``main``, ``report``, ``handoff``,
    ``tool_router``, ``tool_arg_correction``, ``artifact_query``) across the
    grafted parent+children node graph (D5); ``llm_call_count`` keeps counting
    ``main`` calls only, for step semantics.
    """
    prompt_total = completion_total = 0
    llm_count = tool_count = loop_count = anomaly_count = 0
    by_kind: dict[str, dict[str, int]] = {}
    by_agent: dict[str, dict[str, Any]] = {}
    for node in nodes.values():
        if node.type == "anomaly":
            anomaly_count += 1
        if node.type == "llm_call":
            kind = node.attributes.get("kind") or "main"
            prompt = node.attributes.get("prompt_tokens") or 0
            completion = node.attributes.get("completion_tokens") or 0
            prompt_total += prompt
            completion_total += completion
            if kind == "main":
                llm_count += 1
            kind_agg = by_kind.setdefault(
                kind, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            kind_agg["prompt_tokens"] += prompt
            kind_agg["completion_tokens"] += completion
            kind_agg["total_tokens"] += prompt + completion
            agent_agg = by_agent.setdefault(
                node.agent, {"tokens": 0, "call_count": 0, "elapsed_s": None}
            )
            agent_agg["tokens"] += prompt + completion
            agent_agg["call_count"] += 1
        elif node.type == "tool_call":
            tool_count += 1
            if node.loop_flag:
                loop_count += 1
        elif node.type == "session_end":
            agent_agg = by_agent.setdefault(
                node.agent, {"tokens": 0, "call_count": 0, "elapsed_s": None}
            )
            agent_agg["elapsed_s"] = node.attributes.get("elapsed_s")
    return {
        "total_prompt_tokens": prompt_total,
        "total_completion_tokens": completion_total,
        "total_tokens": prompt_total + completion_total,
        "cost_usd": compute_cost(model, prompt_total, completion_total),
        "elapsed_s": end_ev.get("elapsed_s", 0.0) if end_ev else 0.0,
        "llm_call_count": llm_count,
        "tool_call_count": tool_count,
        "loop_count": loop_count,
        "anomaly_count": anomaly_count,
        "by_kind": by_kind,
        "by_agent": by_agent,
    }


# ── Session directory helpers ─────────────────────────────────────────────────


def _summarise_dir(session_dir: Path) -> dict[str, Any] | None:
    """Read the first and last lines of events.jsonl for a lightweight summary.

    Falls back to ``session_data.json`` when ``events.jsonl`` is absent.

    Args:
        session_dir: A single session directory under ``.my_coding_agent/``.

    Returns:
        Summary dict or ``None`` if neither file is readable.
    """
    events_path = session_dir / "events.jsonl"
    if events_path.exists():
        return _summarise_from_events(session_dir.name, events_path)
    data_path = session_dir / "session_data.json"
    if data_path.exists():
        return _summarise_from_data(session_dir.name, data_path)
    return None


def _summarise_from_events(session_id: str, path: Path) -> dict[str, Any] | None:
    """Extract summary fields from the first and last JSONL lines.

    Args:
        session_id: Directory name used as fallback session ID.
        path: Path to ``events.jsonl``.

    Returns:
        Summary dict or ``None`` on parse error.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return None
        first = json.loads(lines[0])
        last = json.loads(lines[-1]) if len(lines) > 1 else {}
        return {
            "session_id": first.get("session_id", session_id),
            "label": first.get("label", "Session"),
            "model": first.get("model", ""),
            "started_at": first.get("started_at", ""),
            "stop_reason": last.get("stop_reason")
            if last.get("type") == "session_end"
            else None,
            "steps": last.get("steps") if last.get("type") == "session_end" else None,
        }
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def _summarise_from_data(session_id: str, path: Path) -> dict[str, Any] | None:
    """Extract summary fields from ``session_data.json``.

    Args:
        session_id: Directory name used as fallback session ID.
        path: Path to ``session_data.json``.

    Returns:
        Summary dict or ``None`` on parse error.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "session_id": data.get("session_id", session_id),
            "label": data.get("label", "Session"),
            "model": data.get("model", ""),
            "started_at": data.get("started_at", ""),
            "stop_reason": data.get("stop_reason"),
            "steps": data.get("steps"),
        }
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def _fallback_session(session_dir: Path, depth: int = 0) -> TraceSession:
    """Build a minimal ``TraceSession`` from ``session_data.json``.

    Used when ``events.jsonl`` does not exist (pre-recorder sessions).

    Args:
        session_dir: Session directory containing ``session_data.json``.
        depth: Call-tree depth of this session's root.

    Returns:
        Two-node session (root + end) marked with ``source=session_data_fallback``.
    """
    session_id = session_dir.name
    data: dict[str, Any] = {}
    data_path = session_dir / "session_data.json"
    if data_path.exists():
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not parse %s: %s", data_path, exc)

    root_id = f"{session_id}::session"
    end_id = f"{session_id}::session_end"
    root = _make_node(
        id=root_id,
        type="session",
        label=data.get("label", "Session"),
        inputs={},
        outputs={},
        attributes={
            "model": data.get("model", ""),
            "started_at": data.get("started_at", ""),
        },
    )

    end = _make_node(
        id=end_id,
        type="session_end",
        label="End",
        inputs={},
        outputs={},
        attributes={
            "stop_reason": data.get("stop_reason", ""),
            "steps": data.get("steps", 0),
        },
        parent_id=root_id,
    )

    root.agent = end.agent = session_id
    root.depth, end.depth = depth, depth + 1

    return TraceSession(
        session_id=session_id,
        label=data.get("label", "Session"),
        model=data.get("model", ""),
        started_at=data.get("started_at", ""),
        ended_at=data.get("ended_at"),
        stop_reason=data.get("stop_reason"),
        steps=data.get("steps", 0),
        nodes={root_id: root, end_id: end},
        order=[root_id, end_id],
        analytics={"source": "session_data_fallback"},
    )


def _stub_session(session_id: str, depth: int = 0) -> TraceSession:
    """Return an empty session stub used to break circular delegate chains.

    Args:
        session_id: The already-visited session ID.
        depth: Call-tree depth of this session's root.

    Returns:
        Minimal ``TraceSession`` with a single root node.
    """
    root_id = f"{session_id}::session"
    root = _make_node(
        id=root_id,
        type="session",
        label="[recursive]",
        inputs={},
        outputs={},
        attributes={"note": "circular reference — skipped"},
    )
    root.agent = session_id
    root.depth = depth
    return TraceSession(
        session_id=session_id,
        label="[recursive]",
        model="",
        started_at="",
        ended_at=None,
        stop_reason=None,
        steps=0,
        nodes={root_id: root},
        order=[root_id],
        analytics={},
    )


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read and parse all JSONL lines from *path*, skipping blank/invalid lines.

    Args:
        path: Path to an ``events.jsonl`` file.

    Returns:
        List of parsed event dicts; invalid lines are silently skipped.
    """
    events: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return events
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)
    return events


def _find_start(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first ``session_start`` event, or an empty dict.

    Args:
        events: Parsed event list.

    Returns:
        The ``session_start`` event dict, or ``{}`` if absent.
    """
    for ev in events:
        if ev.get("type") == "session_start":
            return ev
    return {}


def _find_end(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last ``session_end`` event, or ``None``.

    Args:
        events: Parsed event list.

    Returns:
        The ``session_end`` event dict, or ``None`` if absent.
    """
    for ev in reversed(events):
        if ev.get("type") == "session_end":
            return ev
    return None


def _make_node(
    *,
    id: str,
    type: str,
    label: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    attributes: dict[str, Any],
    parent_id: str | None = None,
) -> TraceNode:
    """Construct a ``TraceNode``.

    Args:
        id: Unique node identifier.
        type: Semantic node type.
        label: Display label.
        inputs: Input data dict.
        outputs: Output data dict.
        attributes: Scalar metadata dict.
        parent_id: Parent node ID or ``None``.

    Returns:
        A new ``TraceNode``.
    """
    return TraceNode(
        id=id,
        type=type,
        label=label,
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        parent_id=parent_id,
        loop_flag=False,
    )

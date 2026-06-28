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

    Args:
        root_id: ID of the session root node; every added node parents to it.
    """

    root_id: str
    nodes: dict[str, TraceNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add_root(self, node: TraceNode) -> None:
        """Register the root node and start the execution order.

        Args:
            node: The session root ``TraceNode``.
        """
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
        return _fallback_session(session_dir)

    events = _read_events(events_path)
    start_ev = _find_start(events)
    session_id = start_ev.get("session_id", session_dir.name)

    if session_id in seen:
        return _stub_session(session_id)
    seen.add(session_id)

    end_ev = _find_end(events)
    steps_groups = _group_into_steps(events)

    root_id = f"{session_id}::session"
    graph = _Graph(root_id=root_id)
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
    _assign_ctx_state(graph.nodes, graph.order, start_ev.get("context_window"))
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

    for ev in group:
        built = _build_event_node(ev, session_id, step, counters)
        if built is None:
            continue
        node_id, node = built
        graph.add(node_id, node, step=step)

        if node.type == "tool_call":
            child_sid = node.attributes.get("child_session_id")
            if child_sid and child_sid not in seen:
                _embed_child_session(child_sid, graph, seen, session_dir)


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
    return node_id, _make_node(
        id=node_id,
        type="llm_call",
        label=label,
        inputs={"messages": ev.get("messages") or []},
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
            "completion_tokens": ev.get("completion"),
            "total_tokens": ev.get("total"),
            "context_window": ev.get("context_window"),
            "started_at": ev.get("started_at", ""),
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
    """Build the ContextPreflightNode node for a ``handoff`` event."""
    node_id = f"{session_id}::step{step}::handoff"
    return node_id, _make_node(
        id=node_id,
        type="handoff",
        label="ContextPreflightNode",
        inputs={},
        outputs={"content": ev.get("content", "")},
        attributes={
            "ctx_tokens": ev.get("ctx_tokens"),
            "ctx_pct": ev.get("ctx_pct"),
            "path": ev.get("path", ""),
            "started_at": ev.get("started_at", ""),
        },
    )


def _build_token_tracking_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the TokenTrackingNode node for a ``token_tracking`` event."""
    node_id = f"{session_id}::step{step}::token_tracking"
    return node_id, _make_node(
        id=node_id,
        type="token_tracking",
        label="TokenTrackingNode",
        inputs={},
        outputs={},
        attributes={
            "prompt_tokens": ev.get("prompt_tokens"),
            "completion_tokens": ev.get("completion_tokens"),
            "total_tokens": ev.get("total_tokens"),
            "ctx_pct": ev.get("ctx_pct"),
            "context_window": ev.get("context_window"),
            "started_at": ev.get("started_at", ""),
        },
    )


def _build_finish_check_node(
    ev: dict[str, Any], session_id: str, step: int, counters: dict[str, int]
) -> tuple[str, TraceNode]:
    """Build the FinishCheckNode node for a ``finish_check`` event."""
    node_id = f"{session_id}::step{step}::finish_check"
    return node_id, _make_node(
        id=node_id,
        type="finish_check",
        label="FinishCheckNode",
        inputs={},
        outputs={},
        attributes={
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
    "token_tracking": _build_token_tracking_node,
    "finish_check": _build_finish_check_node,
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
) -> None:
    """Load a delegate child session and graft its nodes into *graph*.

    The child's nodes are appended to the parent's execution order right after
    the delegating ``tool_call`` node, so the sub-agent's steps appear inline.

    Args:
        child_sid: Session ID of the spawned sub-agent.
        graph: Mutable graph accumulator to extend.
        seen: Recursion guard.
        session_dir: Parent session directory (child is a sibling directory).
    """
    child_dir = session_dir.parent / child_sid
    child_events = child_dir / "events.jsonl"
    child_session = load_session(child_events, _seen=seen)
    graph.nodes.update(child_session.nodes)
    graph.order.extend(child_session.order)


# ── Context-window state ──────────────────────────────────────────────────────


def _node_fill(node: TraceNode) -> int | None:
    """Return the context-window fill (in tokens) this node reports, if any.

    Args:
        node: The trace node to inspect.

    Returns:
        Token count of the context the node operated on, or ``None`` when the
        node carries no token figure (e.g. router, finish_check).
    """
    a = node.attributes
    if node.type in ("llm_call", "token_tracking"):
        return a.get("prompt_tokens")
    if node.type == "handoff":
        return a.get("ctx_tokens")
    return None


def _assign_ctx_state(
    nodes: dict[str, TraceNode],
    order: list[str],
    default_window: int | None,
) -> None:
    """Compute each node's context-window snapshot, in execution order.

    Walks *order* tracking the running context fill.  Nodes that report a token
    figure get a signed ``delta`` versus the previous reporting node — positive
    when the window grew (tokens added), negative after a compaction (tokens
    removed).  Nodes without their own figure inherit the carried fill with a
    zero delta so the UI can still draw the bar.

    Args:
        nodes: All trace nodes; each gains a ``ctx_state`` dict in-place.
        order: Node IDs in execution order.
        default_window: Session context window size used when a node omits it.
    """
    prev = 0
    window = default_window
    for nid in order:
        node = nodes.get(nid)
        if node is None:
            continue
        own_window = node.attributes.get("context_window")
        if own_window:
            window = own_window
        if node.type == "session":
            tokens, delta, measured = 0, 0, True
            prev = 0
        else:
            fill = _node_fill(node)
            if fill is None:
                tokens, delta, measured = prev, 0, False
            else:
                tokens, delta, measured = fill, fill - prev, True
                prev = fill
        pct = round(tokens / window * 100, 1) if window else None
        node.ctx_state = {
            "tokens": tokens,
            "window": window,
            "pct": pct,
            "delta": delta,
            "added": max(delta, 0),
            "removed": max(-delta, 0),
            "measured": measured,
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
        ``llm_call_count``, ``tool_call_count``, ``loop_count``.
    """
    prompt_total = completion_total = 0
    llm_count = tool_count = loop_count = 0
    for node in nodes.values():
        if node.type == "llm_call" and node.attributes.get("kind") == "main":
            prompt_total += node.attributes.get("prompt_tokens") or 0
            completion_total += node.attributes.get("completion_tokens") or 0
            llm_count += 1
        elif node.type == "tool_call":
            tool_count += 1
            if node.loop_flag:
                loop_count += 1
    return {
        "total_prompt_tokens": prompt_total,
        "total_completion_tokens": completion_total,
        "total_tokens": prompt_total + completion_total,
        "cost_usd": compute_cost(model, prompt_total, completion_total),
        "elapsed_s": end_ev.get("elapsed_s", 0.0) if end_ev else 0.0,
        "llm_call_count": llm_count,
        "tool_call_count": tool_count,
        "loop_count": loop_count,
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


def _fallback_session(session_dir: Path) -> TraceSession:
    """Build a minimal ``TraceSession`` from ``session_data.json``.

    Used when ``events.jsonl`` does not exist (pre-recorder sessions).

    Args:
        session_dir: Session directory containing ``session_data.json``.

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


def _stub_session(session_id: str) -> TraceSession:
    """Return an empty session stub used to break circular delegate chains.

    Args:
        session_id: The already-visited session ID.

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

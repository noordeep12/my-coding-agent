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
from pathlib import Path
from typing import Any

from .pricing import compute_cost
from .schema import TraceNode, TraceSession

logger = logging.getLogger(__name__)

# ── Visual constants ────────────────────────────────────────────────────────

_SHAPE: dict[str, str] = {
    "session": "rect",
    "step": "rect",
    "router": "diamond",
    "llm_call": "circle",
    "tool_call": "square",
    "handoff": "rect",
    "session_end": "circle",
}

_COLOR: dict[str, str] = {
    "session": "#4A90D9",
    "step": "#2C3E50",
    "router": "#F39C12",
    "llm_call": "#27AE60",
    "tool_call": "#8E44AD",
    "handoff": "#E74C3C",
    "session_end": "#7F8C8D",
}

# Fixed column x-positions (px)
_COL_X = [60, 200, 340, 480, 620]
_Y_STEP = 90  # px between sibling nodes
_Y_GAP = 60  # extra padding between pipeline steps


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
        Fully parsed ``TraceSession`` with layout coordinates assigned.

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
    nodes: dict[str, TraceNode] = {}
    edges: list[tuple[str, str]] = []

    root_id = f"{session_id}::session"
    nodes[root_id] = _make_node(
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

    prev_id = root_id
    for step_idx, group in enumerate(steps_groups):
        step_id = f"{session_id}::step{step_idx + 1}"
        nodes[step_id] = _make_node(
            id=step_id,
            type="step",
            label=f"Step {step_idx + 1}",
            inputs={},
            outputs={},
            attributes={"step": step_idx + 1},
            parent_id=root_id,
        )
        nodes[root_id].children.append(step_id)
        edges.append((prev_id, step_id))
        prev_id = step_id
        _build_step_nodes(
            group, step_id, session_id, step_idx, nodes, edges, seen, session_dir
        )

    if end_ev:
        end_id = f"{session_id}::session_end"
        nodes[end_id] = _make_node(
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
            parent_id=root_id,
        )
        nodes[root_id].children.append(end_id)
        edges.append((prev_id, end_id))

    model = start_ev.get("model", "")
    _layout(nodes, root_id)
    _detect_loops(nodes)
    analytics = _compute_analytics(nodes, model, end_ev)

    canvas_w, canvas_h = _canvas_size(nodes)
    return TraceSession(
        session_id=session_id,
        label=start_ev.get("label", "Session"),
        model=model,
        started_at=start_ev.get("started_at", ""),
        ended_at=end_ev.get("ended_at") if end_ev else None,
        stop_reason=end_ev.get("stop_reason") if end_ev else None,
        steps=len(steps_groups),
        nodes=nodes,
        edges=edges,
        analytics=analytics,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
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
    step_id: str,
    session_id: str,
    step_idx: int,
    nodes: dict[str, TraceNode],
    edges: list[tuple[str, str]],
    seen: set[str],
    session_dir: Path,
) -> None:
    """Add child nodes for one pipeline step into *nodes* and *edges*.

    Args:
        group: All events belonging to this step (starts with ``router``).
        step_id: Parent step node ID.
        session_id: Owning session ID (used for stable node IDs).
        step_idx: Zero-based step index.
        nodes: Mutable node dict to populate.
        edges: Mutable edge list to populate.
        seen: Recursion guard set for delegate sessions.
        session_dir: Filesystem directory of the owning session.
    """
    prev_id = step_id
    llm_counter = 0
    tool_counter = 0

    for ev in group:
        ev_type = ev.get("type", "")

        if ev_type == "router":
            node_id = f"{session_id}::step{step_idx + 1}::router"
            nodes[node_id] = _make_node(
                id=node_id,
                type="router",
                label="Router",
                inputs={"signal": ev.get("signal", "")[:120]},
                outputs={"selected": ev.get("selected", [])},
                attributes={
                    "phase": ev.get("phase", ""),
                    "used_llm": ev.get("used_llm", False),
                    "started_at": ev.get("started_at", ""),
                },
                parent_id=step_id,
            )
            nodes[step_id].children.append(node_id)
            edges.append((prev_id, node_id))
            prev_id = node_id

        elif ev_type == "llm_call":
            llm_counter += 1
            kind = ev.get("kind", "main")
            node_id = f"{session_id}::step{step_idx + 1}::llm::{llm_counter}"
            label = f"LLM #{ev.get('call', llm_counter)}"
            if kind != "main":
                label += f" ({kind})"
            resp = ev.get("response") or {}
            nodes[node_id] = _make_node(
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
                parent_id=step_id,
            )
            nodes[step_id].children.append(node_id)
            edges.append((prev_id, node_id))
            prev_id = node_id

        elif ev_type == "tool_call":
            tool_counter += 1
            node_id = f"{session_id}::step{step_idx + 1}::tool::{tool_counter}"
            child_sid = ev.get("child_session_id")
            label = ev.get("name", "tool")
            nodes[node_id] = _make_node(
                id=node_id,
                type="tool_call",
                label=label,
                inputs={"args": ev.get("args", {})},
                outputs={"result": ev.get("result", "")},
                attributes={
                    "name": ev.get("name", ""),
                    "latency_s": ev.get("latency_s"),
                    "started_at": ev.get("started_at", ""),
                    "child_session_id": child_sid,
                },
                parent_id=step_id,
            )
            nodes[step_id].children.append(node_id)
            edges.append((prev_id, node_id))
            prev_id = node_id

            if child_sid and child_sid not in seen:
                _embed_child_session(
                    child_sid, node_id, nodes, edges, seen, session_dir
                )

        elif ev_type == "handoff":
            node_id = f"{session_id}::step{step_idx + 1}::handoff"
            nodes[node_id] = _make_node(
                id=node_id,
                type="handoff",
                label="Handoff",
                inputs={},
                outputs={"content": ev.get("content", "")},
                attributes={
                    "step": ev.get("step"),
                    "ctx_tokens": ev.get("ctx_tokens"),
                    "ctx_pct": ev.get("ctx_pct"),
                    "path": ev.get("path", ""),
                    "started_at": ev.get("started_at", ""),
                },
                parent_id=step_id,
            )
            nodes[step_id].children.append(node_id)
            edges.append((prev_id, node_id))
            prev_id = node_id

        elif ev_type == "token_tracking":
            nodes[step_id].attributes.update(
                {
                    "prompt_tokens": ev.get("prompt_tokens"),
                    "completion_tokens": ev.get("completion_tokens"),
                    "total_tokens": ev.get("total_tokens"),
                    "ctx_pct": ev.get("ctx_pct"),
                    "context_window": ev.get("context_window"),
                }
            )

        elif ev_type == "finish_check":
            nodes[step_id].attributes.update(
                {
                    "finish_reason": ev.get("finish_reason"),
                    "signal": ev.get("signal"),
                }
            )


def _embed_child_session(
    child_sid: str,
    parent_node_id: str,
    nodes: dict[str, TraceNode],
    edges: list[tuple[str, str]],
    seen: set[str],
    session_dir: Path,
) -> None:
    """Load a delegate child session and graft its nodes into *nodes*.

    Args:
        child_sid: Session ID of the spawned sub-agent.
        parent_node_id: The ``delegate`` tool_call node that spawned it.
        nodes: Mutable node dict to extend.
        edges: Mutable edge list to extend.
        seen: Recursion guard.
        session_dir: Parent session directory (child is a sibling directory).
    """
    child_dir = session_dir.parent / child_sid
    child_events = child_dir / "events.jsonl"
    child_session = load_session(child_events, _seen=seen)
    nodes.update(child_session.nodes)
    edges.extend(child_session.edges)
    child_root = f"{child_sid}::session"
    if child_root in nodes:
        nodes[parent_node_id].children.append(child_root)
        edges.append((parent_node_id, child_root))


# ── Layout ────────────────────────────────────────────────────────────────────


def _layout(nodes: dict[str, TraceNode], root_id: str) -> None:
    """Assign ``(x, y)`` pixel coordinates to every node in *nodes* in-place.

    Uses a depth-first pre-order walk.  Each node's x is determined by its
    depth in the tree; its y is the midpoint of its children's y-range (or the
    next available leaf y for leaf nodes).

    Args:
        nodes: All nodes; modified in-place.
        root_id: ID of the root node to start the walk from.
    """
    counter = [0]

    def _walk(node_id: str, depth: int) -> int:
        node = nodes.get(node_id)
        if node is None:
            return counter[0] * _Y_STEP + 60
        col = min(depth, len(_COL_X) - 1)
        node.x = _COL_X[col]
        if not node.children:
            y = counter[0] * _Y_STEP + 60
            counter[0] += 1
            node.y = y
            return y
        child_ys = [_walk(c, depth + 1) for c in node.children]
        node.y = (child_ys[0] + child_ys[-1]) // 2
        if node.type == "step" and counter[0] > 0:
            node.y += _Y_GAP // 2
        return node.y

    _walk(root_id, 0)


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
    root.x, root.y = _COL_X[0], 60
    root.children = [end_id]

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
    end.x, end.y = _COL_X[1], 60

    return TraceSession(
        session_id=session_id,
        label=data.get("label", "Session"),
        model=data.get("model", ""),
        started_at=data.get("started_at", ""),
        ended_at=data.get("ended_at"),
        stop_reason=data.get("stop_reason"),
        steps=data.get("steps", 0),
        nodes={root_id: root, end_id: end},
        edges=[(root_id, end_id)],
        analytics={"source": "session_data_fallback"},
        canvas_width=_COL_X[1] + 100,
        canvas_height=160,
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
    root.x, root.y = _COL_X[0], 60
    return TraceSession(
        session_id=session_id,
        label="[recursive]",
        model="",
        started_at="",
        ended_at=None,
        stop_reason=None,
        steps=0,
        nodes={root_id: root},
        edges=[],
        analytics={},
        canvas_width=160,
        canvas_height=120,
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
    """Construct a ``TraceNode`` with shape/color defaults applied.

    Args:
        id: Unique node identifier.
        type: Semantic node type.
        label: Display label.
        inputs: Input data dict.
        outputs: Output data dict.
        attributes: Scalar metadata dict.
        parent_id: Parent node ID or ``None``.

    Returns:
        A new ``TraceNode`` with ``x=0``, ``y=0`` (layout assigns coords later).
    """
    return TraceNode(
        id=id,
        type=type,
        label=label,
        shape=_SHAPE.get(type, "rect"),
        color=_COLOR.get(type, "#555"),
        x=0,
        y=0,
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        children=[],
        parent_id=parent_id,
        loop_flag=False,
    )


def _canvas_size(nodes: dict[str, TraceNode]) -> tuple[int, int]:
    """Compute the bounding box of all node positions.

    Args:
        nodes: All positioned nodes.

    Returns:
        ``(width, height)`` with 120px padding on each axis.
    """
    if not nodes:
        return 700, 200
    max_x = max(n.x for n in nodes.values())
    max_y = max(n.y for n in nodes.values())
    return max_x + 120, max_y + 120

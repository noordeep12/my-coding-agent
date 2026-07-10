"""Read-only descriptor of the framework's registered pipeline node types.

Introspects `nodes/*` (the actual step nodes `build_default_pipeline` wires
into `Pipeline`) so a consumer — the webui builder — can offer exactly the
node types the framework registers, in their canonical execution order. This
is a thin registry/introspection helper, not a new node system: no node type
is defined here, and no node behavior is altered.
"""

from __future__ import annotations

from dataclasses import dataclass

from .nodes import (
    AnomalyDetectNode,
    ContextGuardNode,
    FinalizeStepNode,
    LLMCallNode,
    ToolDispatchNode,
)


@dataclass(frozen=True)
class NodeTypeDescriptor:
    """One registered node type: its name and its editable option schema.

    `options` is empty for every node type today — none of the registered
    nodes take user-facing configuration (their constructor args, where
    present, are runtime wiring such as callables supplied by `AgentNode`,
    not data a builder UI could serialize). The schema is still exposed
    per-type so a future node with real options needs no API change.
    """

    name: str
    options: tuple[dict[str, str], ...] = ()


#: The registered node types, in the canonical order `build_default_pipeline`
#: wires them into `Pipeline`. `ContextSummarizerNode` and `ToolRoutingNode`
#: are registered node types (importable from `pipeline.nodes`) but are not
#: part of the default step sequence — nested helpers driven by other nodes,
#: not independent stages a graph places directly — so they are excluded from
#: the placeable set.
NODE_TYPES: tuple[NodeTypeDescriptor, ...] = (
    NodeTypeDescriptor(ContextGuardNode.name),
    NodeTypeDescriptor(LLMCallNode.name),
    NodeTypeDescriptor(ToolDispatchNode.name),
    NodeTypeDescriptor(AnomalyDetectNode.name),
    NodeTypeDescriptor(FinalizeStepNode.name),
)

#: Canonical execution order (by node type name) — the only order
#: `build_default_pipeline`/`AgentNode` can actually execute today.
CANONICAL_ORDER: tuple[str, ...] = tuple(n.name for n in NODE_TYPES)


def list_node_types() -> list[dict[str, object]]:
    """Return the placeable node types as JSON-serializable dicts."""
    return [
        {"name": descriptor.name, "options": list(descriptor.options)}
        for descriptor in NODE_TYPES
    ]


def _walk_chain(
    by_id: dict[str, dict[str, object]],
    outgoing: dict[str, list[str]],
    start: str,
    end: str,
) -> tuple[list[str], str] | str:
    """Walk the single start→end chain; return `(node_ids, "")` or an error string."""
    visited_ids: list[str] = []
    seen_ids: set[str] = set()
    current: str | None = start
    while current is not None:
        if current in seen_ids:
            return "graph contains a cycle"
        seen_ids.add(current)
        visited_ids.append(current)
        if current == end:
            return visited_ids, ""
        nexts = outgoing.get(current, [])
        if len(nexts) != 1:
            return f"node {current!r} must have exactly one outgoing link to reach end"
        current = nexts[0]
        if current not in by_id:
            return "dangling link: edge points to a node not in the graph"
    return "graph has no path from start to end"


def validate_runnable(
    nodes: list[dict[str, object]],
    edges: list[dict[str, str]],
    start: str | None,
    end: str | None,
) -> str | None:
    """Return ``None`` if the graph is runnable, else a short error message.

    A graph is runnable when it is a single chain (no branching, no orphan
    nodes) from `start` to `end` whose node types, walked in edge order,
    equal `CANONICAL_ORDER` exactly — the only sequence `AgentNode` can
    actually execute today (D3/D5: composition is from registered types
    only; this increment does not support reordering the fixed step loop).
    """
    if not start or not end:
        return "graph has no start/end designation"

    by_id: dict[str, dict[str, object]] = {str(n["id"]): n for n in nodes}
    if start not in by_id or end not in by_id:
        return "start/end must reference nodes in the graph"

    valid_types = {d.name for d in NODE_TYPES}
    for node in nodes:
        if node.get("type") not in valid_types:
            return f"unregistered node type: {node.get('type')!r}"

    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge["from"], []).append(edge["to"])

    result = _walk_chain(by_id, outgoing, start, end)
    if isinstance(result, str):
        return result
    visited_ids, _ = result

    if len(visited_ids) != len(nodes):
        return "graph has nodes not connected on the start→end path"

    visited_types = tuple(str(by_id[node_id]["type"]) for node_id in visited_ids)
    if visited_types != CANONICAL_ORDER:
        return "node order must match the registered execution order: " + " → ".join(
            CANONICAL_ORDER
        )

    return None

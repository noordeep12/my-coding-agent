"""Typed contracts for the Trace Explorer.

``TraceNode`` and ``TraceSession`` are the data shapes produced by ``reader.py``
and consumed by ``server.py``.  Keeping them here follows the project convention
of one ``schema.py`` per package (CONTRIBUTE.md §25).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceNode:
    """One node in the trace graph.

    Args:
        id: Unique stable identifier, e.g. ``"s0::step1::llm::0"``.
        type: Semantic type — one of ``session``, ``step``, ``router``,
            ``llm_call``, ``tool_call``, ``handoff``, ``session_end``.
        label: Human-readable display name shown beneath the node shape.
        shape: SVG shape hint — ``diamond``, ``circle``, ``square``, ``rect``.
        color: Hex fill colour for the SVG shape.
        x: Horizontal centre position in SVG canvas pixels.
        y: Vertical centre position in SVG canvas pixels.
        inputs: Structured inputs for this event (prompt messages, tool args, …).
        outputs: Structured outputs (LLM response, tool result, …).
        attributes: Scalar metadata (latency_s, tokens, cost_usd, phase, …).
        children: Ordered list of child node IDs.
        parent_id: Parent node ID, or ``None`` for the root.
        loop_flag: ``True`` when this tool_call repeats an earlier (name, args)
            combination — signals a potential agent loop.
    """

    id: str
    type: str
    label: str
    shape: str
    color: str
    x: int
    y: int
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    parent_id: str | None = None
    loop_flag: bool = False


@dataclass
class TraceSession:
    """Full parsed session ready for the Trace Explorer UI.

    Args:
        session_id: The session's unique identifier.
        label: Agent label from the ``session_start`` event.
        model: Model name used in this session.
        started_at: ISO-8601 start timestamp.
        ended_at: ISO-8601 end timestamp, or ``None`` if the session is ongoing.
        stop_reason: Final stop reason (``max_steps``, ``stop``, …), or ``None``.
        steps: Number of pipeline steps executed.
        nodes: All ``TraceNode`` instances keyed by their ``id``.
        edges: Ordered list of ``(from_id, to_id)`` pairs for SVG path rendering.
        analytics: Aggregate statistics — total_tokens, cost_usd, elapsed_s,
            loop_count, llm_call_count, tool_call_count.
        canvas_width: Total SVG canvas width in pixels.
        canvas_height: Total SVG canvas height in pixels.
    """

    session_id: str
    label: str
    model: str
    started_at: str
    ended_at: str | None
    stop_reason: str | None
    steps: int
    nodes: dict[str, TraceNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    analytics: dict[str, Any] = field(default_factory=dict)
    canvas_width: int = 700
    canvas_height: int = 200

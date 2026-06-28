"""Typed contracts for the Trace Explorer.

``TraceNode`` and ``TraceSession`` are the data shapes produced by ``reader.py``
and consumed by ``server.py``.  Keeping them here follows the project convention
of one ``schema.py`` per package (CONTRIBUTE.md §25).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceNode:
    """One node in the trace — a single pipeline stage that processed RunContext.

    Args:
        id: Unique stable identifier, e.g. ``"s0::step1::llm::1"``.
        type: Semantic type — one of ``session``, ``router``, ``llm_call``,
            ``tool_call``, ``handoff``, ``token_tracking``, ``finish_check``,
            ``session_end``.
        label: Human-readable display name (the pipeline node's class name).
        inputs: Structured inputs for this event (prompt messages, tool args, …).
        outputs: Structured outputs (LLM response, tool result, …).
        attributes: Scalar metadata (step, latency_s, tokens, phase, …).
        parent_id: Parent (session root) node ID, or ``None`` for the root.
        loop_flag: ``True`` when this tool_call repeats an earlier (name, args)
            combination — signals a potential agent loop.
        ctx_state: Per-node snapshot of the session context window after this
            node ran. Keys: ``tokens`` (current fill), ``window`` (max size),
            ``pct`` (fill percentage), ``added``/``removed`` (tokens grown or
            compacted at this node), ``delta`` (signed net change), ``measured``
            (``True`` when the node carries its own token figure).
    """

    id: str
    type: str
    label: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    loop_flag: bool = False
    ctx_state: dict[str, Any] = field(default_factory=dict)


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
        order: Node IDs in execution order — the linear spine the UI walks for
            keyboard navigation, the Explorer rail, and the Tree view.
        analytics: Aggregate statistics — total_tokens, cost_usd, elapsed_s,
            loop_count, llm_call_count, tool_call_count.
    """

    session_id: str
    label: str
    model: str
    started_at: str
    ended_at: str | None
    stop_reason: str | None
    steps: int
    nodes: dict[str, TraceNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    analytics: dict[str, Any] = field(default_factory=dict)


# Signature shared by every per-event-type node builder in ``reader.py``:
# ``(event, session_id, step, counters) -> (node_id, node)``.
EventBuilder = Callable[
    [dict[str, Any], str, int, dict[str, int]], tuple[str, TraceNode]
]

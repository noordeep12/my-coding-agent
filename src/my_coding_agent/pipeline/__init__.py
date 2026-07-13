"""Pipeline package — node-based DAG execution engine for the agentic loop."""

from __future__ import annotations

from typing import Callable

from .context import RunContext
from .dag import Pipeline
from .handoff import handoff_to_user_message, save_handoff
from .node import BaseNode, Node
from .nodes import (
    AnomalyDetectNode,
    ContextGuardNode,
    FinalizeStepNode,
    LLMCallNode,
    ToolDispatchNode,
)
from .schema import ContextHandoff

__all__ = [
    "RunContext",
    "Pipeline",
    "Node",
    "BaseNode",
    "ContextHandoff",
    "handoff_to_user_message",
    "save_handoff",
    "AnomalyDetectNode",
    "ContextGuardNode",
    "FinalizeStepNode",
    "LLMCallNode",
    "ToolDispatchNode",
    "AgentNode",
    "DEFAULT_MAX_STEPS",
    "build_default_pipeline",
]


def __getattr__(name: str):
    # Lazy re-export: AgentNode (pipeline.nodes.agent) imports build_default_pipeline
    # from this package at call time, so an eager import here would cycle.
    if name in ("AgentNode", "DEFAULT_MAX_STEPS"):
        from .nodes.agent import DEFAULT_MAX_STEPS, AgentNode

        return {"AgentNode": AgentNode, "DEFAULT_MAX_STEPS": DEFAULT_MAX_STEPS}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_default_pipeline(
    spawn_fn: Callable[[], list[dict]] | None = None,
    checkpoint_fn: Callable[[RunContext], None] | None = None,
) -> Pipeline:
    """Return a Pipeline with the standard 5-node agentic loop.

    Args:
        spawn_fn: Optional callable passed to ``ContextGuardNode`` for
            spawning continuation agents on context reset.  ``Agent`` passes
            ``self._spawn_continuation`` here to avoid a circular import between
            this package and ``agent.py``.
        checkpoint_fn: Optional callable invoked after each completed step to
            persist the resume checkpoint (run-resilience D3).  ``AgentNode``
            passes its own atomic checkpoint writer here.
    """
    return Pipeline(
        [
            ContextGuardNode(spawn_fn=spawn_fn),
            LLMCallNode(),
            ToolDispatchNode(),
            AnomalyDetectNode(),
            FinalizeStepNode(),
        ],
        checkpoint_fn=checkpoint_fn,
    )

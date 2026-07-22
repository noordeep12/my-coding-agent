"""Pipeline package — node-based workflow graph engine for the agentic loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .context import RunContext
from .graph import Pipeline
from .handoff import handoff_to_user_message, save_handoff
from .node import BaseNode, Node
from .nodes import (
    AnomalyDetectNode,
    ContextGuardNode,
    FinalizeStepNode,
    LLMCallNode,
    PromptStageNode,
    ToolDispatchNode,
)
from .schema import ContextHandoff, Transition

__all__ = [
    "DEFAULT_MAX_STEPS",
    "AgentNode",
    "AnomalyDetectNode",
    "BaseNode",
    "ContextGuardNode",
    "ContextHandoff",
    "FinalizeStepNode",
    "LLMCallNode",
    "Node",
    "Pipeline",
    "PromptStageNode",
    "RunContext",
    "ToolDispatchNode",
    "Transition",
    "build_default_pipeline",
    "handoff_to_user_message",
    "save_handoff",
]


def __getattr__(name: str) -> Any:
    # Lazy re-export: AgentNode (pipeline.nodes.agent) imports build_default_pipeline
    # from this package at call time, so an eager import here would cycle.
    if name in ("AgentNode", "DEFAULT_MAX_STEPS"):
        from .nodes.agent import DEFAULT_MAX_STEPS, AgentNode

        return {"AgentNode": AgentNode, "DEFAULT_MAX_STEPS": DEFAULT_MAX_STEPS}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_default_pipeline(
    spawn_fn: Callable[[], list[dict[str, Any]]] | None = None,
    checkpoint_fn: Callable[[RunContext], None] | None = None,
    transitions: list[Transition] | None = None,
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
        transitions: Optional declared conditional transitions between the
            default nodes (issue #228). ``None`` (the default) leaves the
            five-node loop exactly as it was before this capability existed.
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
        transitions=transitions,
    )

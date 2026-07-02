"""Pipeline package — node-based DAG execution engine for the agentic loop."""

from __future__ import annotations

from typing import Callable

from ..engine.routing import ToolRouter
from .context import RunContext
from .dag import Pipeline
from .node import BaseNode, Node
from .nodes import (
    ContextGuardNode,
    FinalizeStepNode,
    LLMCallNode,
    ToolDispatchNode,
    ToolRoutingNode,
)
from .schema import ContextHandoff

__all__ = [
    "RunContext",
    "Pipeline",
    "Node",
    "BaseNode",
    "ContextHandoff",
    "ToolRouter",
    "ContextGuardNode",
    "FinalizeStepNode",
    "LLMCallNode",
    "ToolDispatchNode",
    "ToolRoutingNode",
    "build_default_pipeline",
]


def build_default_pipeline(
    spawn_fn: Callable[[], list[dict]] | None = None,
) -> Pipeline:
    """Return a Pipeline with the standard 5-node agentic loop.

    Args:
        spawn_fn: Optional callable passed to ``ContextGuardNode`` for
            spawning continuation agents on context reset.  ``Agent`` passes
            ``self._spawn_continuation`` here to avoid a circular import between
            this package and ``agent.py``.
    """
    return Pipeline(
        [
            ContextGuardNode(spawn_fn=spawn_fn),
            ToolRoutingNode(),
            LLMCallNode(),
            ToolDispatchNode(),
            FinalizeStepNode(),
        ]
    )

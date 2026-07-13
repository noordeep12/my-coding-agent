"""Pipeline — ordered sequence of nodes and the step-loop execution engine."""

from __future__ import annotations

from collections.abc import Callable

from .context import RunContext
from .node import Node


class Pipeline:
    """Run an ordered list of nodes through the agentic step loop.

    ``run_step`` executes every node in order for one step, stopping early when
    a node sets ``ctx.signal`` to anything other than ``"CONTINUE"``.
    ``execute`` wraps that in the outer ``while`` loop, handling ``STOP`` and
    ``RESET`` signals.
    """

    def __init__(
        self,
        step_nodes: list[Node],
        checkpoint_fn: Callable[[RunContext], None] | None = None,
    ) -> None:
        self._step_nodes = step_nodes
        # Called by the engine after each *completed* step to persist the resume
        # checkpoint (run-resilience D3). A step that raises never reaches it, so
        # a partial step is discarded (D4). None → no checkpointing.
        self._checkpoint_fn = checkpoint_fn

    def run_step(self, ctx: RunContext) -> None:
        """Run all nodes for one step; short-circuit on non-CONTINUE signal."""
        for node in self._step_nodes:
            node.run(ctx)
            if ctx.signal != "CONTINUE":
                return

    def execute(self, ctx: RunContext) -> list[dict]:
        """Drive the step loop until a terminal signal or max_steps."""
        ctx.signal = "CONTINUE"
        while ctx.step_num < ctx.max_steps and ctx.signal == "CONTINUE":
            self.run_step(ctx)
            # A RESET hands control to the continuation, which ran to completion
            # inside run_step and owns the resumable checkpoint; writing a fresh
            # main checkpoint here would give the pre-reset run a newer mtime and
            # mistarget --resume-last. So return without checkpointing this step.
            if ctx.signal == "RESET":
                return ctx.continuation_messages
            # The step completed (no exception): persist the resume checkpoint at
            # this boundary before acting on the signal (D3/D4).
            if self._checkpoint_fn is not None:
                self._checkpoint_fn(ctx)
            if ctx.signal == "STOP":
                break
        return ctx.messages

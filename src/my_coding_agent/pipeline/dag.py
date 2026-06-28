"""Pipeline — ordered sequence of nodes and the step-loop execution engine."""

from __future__ import annotations

from .context import RunContext
from .node import Node


class Pipeline:
    """Run an ordered list of nodes through the agentic step loop.

    ``run_step`` executes every node in order for one step, stopping early when
    a node sets ``ctx.signal`` to anything other than ``"CONTINUE"``.
    ``execute`` wraps that in the outer ``while`` loop, handling ``STOP`` and
    ``RESET`` signals.
    """

    def __init__(self, step_nodes: list[Node]) -> None:
        self._step_nodes = step_nodes

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
            if ctx.signal == "RESET":
                return ctx.continuation_messages
            if ctx.signal == "STOP":
                break
        return ctx.messages

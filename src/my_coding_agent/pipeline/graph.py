"""Pipeline — workflow graph of nodes and the step-loop execution engine.

Nodes normally run in fixed order (the "CONTINUE" path); a node may instead
set ``ctx.signal = "JUMP"`` with ``ctx.jump_target`` to redirect execution to
another declared node (see ``pipeline.schema.Transition``). Every backward
transition — the only way to form a cycle — carries a hard per-loop round
ceiling, validated at construction and enforced at runtime, so every loop the
graph can express is guaranteed to terminate (issue #228).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..utils import get_logger
from .context import RunContext
from .node import Node
from .schema import Transition

_logger = get_logger(__name__)


class Pipeline:
    """Run a workflow graph of nodes through the agentic step loop.

    ``run_step`` executes nodes in order starting at a given index for one
    pass, stopping early when a node sets ``ctx.signal`` to anything other
    than ``"CONTINUE"``. ``execute`` wraps that in the outer ``while`` loop,
    handling ``STOP``, ``RESET``, and ``JUMP`` signals.

    A pipeline built without ``transitions`` behaves exactly as the original
    fixed-order graph: no node may ever set ``JUMP`` and win, since no
    (source, target) pair is declared (design D1 — byte-identical for
    loop-free pipelines).
    """

    def __init__(
        self,
        step_nodes: list[Node],
        checkpoint_fn: Callable[[RunContext], None] | None = None,
        transitions: list[Transition] | None = None,
    ) -> None:
        self._step_nodes = step_nodes
        # Called by the engine after each *completed* step to persist the resume
        # checkpoint (run-resilience D3). A step that raises never reaches it, so
        # a partial step is discarded (D4). None → no checkpointing.
        self._checkpoint_fn = checkpoint_fn
        self._transitions = list(transitions or [])
        self._index_by_name = {node.name: idx for idx, node in enumerate(step_nodes)}
        self._transitions_by_pair: dict[tuple[str, str], Transition] = {}
        self._validate_transitions()

    def _validate_transitions(self) -> None:
        """Reject unknown node names and unbounded backward transitions (D3).

        Runs at construction time, before any step executes, so a
        misconfigured workflow graph never gets a chance to run.
        """
        for t in self._transitions:
            if t.source not in self._index_by_name:
                raise ValueError(
                    f"Transition source {t.source!r} is not a node in this pipeline"
                )
            if t.target not in self._index_by_name:
                raise ValueError(
                    f"Transition target {t.target!r} is not a node in this pipeline"
                )
            is_backward = self._index_by_name[t.target] <= self._index_by_name[t.source]
            if is_backward and not t.max_rounds:
                raise ValueError(
                    f"Backward transition {t.source!r}->{t.target!r} must declare "
                    "max_rounds (every loop must have a bound)"
                )
            self._transitions_by_pair[(t.source, t.target)] = t

    def run_step(self, ctx: RunContext, start_idx: int = 0) -> int:
        """Run nodes from *start_idx*; short-circuit on non-CONTINUE signal.

        Returns the index of the node that produced the terminal signal, or
        ``len(step_nodes)`` when the whole pass completed with ``CONTINUE``.
        """
        idx = start_idx
        while idx < len(self._step_nodes):
            node = self._step_nodes[idx]
            node.run(ctx)
            if ctx.signal != "CONTINUE":
                return idx
            idx += 1
        return idx

    def _resolve_jump(self, ctx: RunContext, source_name: str) -> int:
        """Validate and apply a JUMP taken by *source_name*; return the next index.

        Increments the round counter for a backward transition and stops the
        run with a distinct ``loop_bound:`` reason once its ceiling is
        reached (D4); otherwise records the taken transition and resumes at
        the target node.
        """
        target_name = ctx.jump_target
        key = (source_name, target_name)
        transition = self._transitions_by_pair.get(key)
        if transition is None:
            raise ValueError(
                f"Undeclared transition {source_name!r}->{target_name!r}: JUMP "
                "may only target a pair declared on this pipeline"
            )

        round_num = 0
        if transition.max_rounds is not None:
            counter_key = f"{source_name}->{target_name}"
            current = ctx.round_counters.get(counter_key, 0)
            if current >= transition.max_rounds:
                stop_reason = f"loop_bound:{source_name}->{target_name}"
                ctx.stop_reason = stop_reason
                ctx.signal = "STOP"
                ctx.recorder.record_transition(
                    source=source_name,
                    target=target_name,
                    round_num=current,
                    outcome="bound_exhausted",
                    step=ctx.step_num,
                )
                _logger.warning(
                    "Loop bound exhausted: %s (round %d/%d)",
                    stop_reason,
                    current,
                    transition.max_rounds,
                )
                return self._index_by_name[source_name]
            round_num = current + 1
            ctx.round_counters[counter_key] = round_num

        ctx.recorder.record_transition(
            source=source_name,
            target=target_name,
            round_num=round_num,
            outcome="jump",
            step=ctx.step_num,
        )
        ctx.signal = "CONTINUE"
        ctx.jump_target = ""
        return self._index_by_name[target_name]

    def execute(self, ctx: RunContext) -> list[dict[str, Any]]:
        """Drive the step loop until a terminal signal or max_steps."""
        ctx.signal = "CONTINUE"
        idx = 0
        while ctx.step_num < ctx.max_steps and ctx.signal == "CONTINUE":
            stop_idx = self.run_step(ctx, idx)
            if ctx.signal == "JUMP":
                source_name = self._step_nodes[stop_idx].name
                idx = self._resolve_jump(ctx, source_name)
                continue
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
            idx = 0
        return ctx.messages

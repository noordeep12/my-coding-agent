"""ContextPreflightNode — check context usage before each step."""

from __future__ import annotations

import json
from typing import Any, Callable

from ...logger import get_logger
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class ContextPreflightNode(BaseNode):
    """Check the context-window usage ratio and signal STOP or RESET as needed.

    When the ratio hits 100 % the node sets signal=STOP (context exhausted).
    When it crosses context_reset_threshold it calls spawn_fn to produce a
    continuation agent, stores the result in ctx.continuation_messages, and
    sets signal=RESET so the pipeline engine returns those messages.

    Args:
        spawn_fn: Callable[[], list[dict]] that generates and runs the
            continuation agent.  Provided by Agent to avoid a circular import.
            When None (tests / custom pipelines) the reset path is skipped and
            the node sets signal=STOP instead.
        t_start: monotonic start time for elapsed-seconds accounting on reset.
    """

    name = "context_preflight"

    def __init__(
        self,
        spawn_fn: Callable[[], list[dict[str, Any]]] | None = None,
        t_start: float | None = None,
    ) -> None:
        self._spawn_fn = spawn_fn
        self._t_start = t_start

    def run(self, ctx: RunContext) -> None:
        if not ctx.llm.context_window:
            ctx.signal = "CONTINUE"
            return

        ctx_tokens = ctx.last_prompt_tokens or len(json.dumps(ctx.messages)) // 2
        ctx_pct = ctx_tokens / ctx.llm.context_window

        if ctx_pct >= 1.0:
            ctx.stop_reason = "context_limit"
            _logger.warning(
                "Context limit reached: %d / %d tokens (%.1f%%). Stopping.",
                ctx_tokens,
                ctx.llm.context_window,
                ctx_pct * 100,
            )
            ctx.signal = "STOP"
            return

        if ctx_pct >= ctx.context_reset_threshold:
            _logger.warning(
                "Context reset threshold reached: %.1f%% used (%d / %d tokens). "
                "Generating handoff and spawning continuation.",
                ctx_pct * 100,
                ctx_tokens,
                ctx.llm.context_window,
            )
            if self._spawn_fn is None:
                # No spawn callable — treat as a hard stop (e.g. in tests).
                ctx.stop_reason = "context_limit"
                ctx.signal = "STOP"
                return

            ctx.continuation_messages = self._spawn_fn()
            ctx.signal = "RESET"
            return

        if ctx_pct >= 0.6:
            _logger.warning(
                "Context at %.1f%% (%d / %d tokens) — reset at %.0f%%.",
                ctx_pct * 100,
                ctx_tokens,
                ctx.llm.context_window,
                ctx.context_reset_threshold * 100,
            )

        ctx.signal = "CONTINUE"

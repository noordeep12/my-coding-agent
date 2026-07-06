"""ContextGuardNode — guard each step on the context-window budget."""

from __future__ import annotations

import json
from typing import Any, Callable

from ...engine.llm.schema import CALL_KIND_HANDOFF, CALL_KIND_REPORT
from ...engine.tool_registry import artifact_file_path
from ...utils import get_logger
from ..context import RunContext
from ..node import BaseNode
from ..supersession import build_stub, find_retirements, supersession_enabled
from .context_summarizer import ContextSummarizerNode

_logger = get_logger(__name__)


class ContextGuardNode(BaseNode):
    """Check the context-window usage ratio and signal STOP or RESET as needed.

    When the ratio hits 100 % the node sets signal=STOP (context exhausted),
    first triggering ``ContextSummarizerNode`` for the hand-back report when
    the run owes one to a delegating parent. When it crosses
    context_reset_threshold it triggers ``ContextSummarizerNode`` for the
    handoff summary (→ ``ctx.handoff_content``), then calls spawn_fn to produce
    a continuation agent, stores the result in ctx.continuation_messages, and
    sets signal=RESET so the pipeline engine returns those messages.

    Args:
        spawn_fn: Callable[[], list[dict]] that generates and runs the
            continuation agent.  Provided by Agent to avoid a circular import.
            When None (tests / custom pipelines) the reset path is skipped and
            the node sets signal=STOP instead.
        t_start: monotonic start time for elapsed-seconds accounting on reset.
    """

    name = "context_guard"

    def __init__(
        self,
        spawn_fn: Callable[[], list[dict[str, Any]]] | None = None,
        t_start: float | None = None,
    ) -> None:
        self._spawn_fn = spawn_fn
        self._t_start = t_start
        self._handoff_summarizer = ContextSummarizerNode(
            kind=CALL_KIND_HANDOFF, triggered_by=self.name
        )
        self._report_summarizer = ContextSummarizerNode(
            kind=CALL_KIND_REPORT, triggered_by=self.name
        )

    def run(self, ctx: RunContext) -> None:
        self._retire_superseded_results(ctx)

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
            if ctx.needs_handback:
                self._report_summarizer.run(ctx)
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

            self._handoff_summarizer.run(ctx)
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

    def _retire_superseded_results(self, ctx: RunContext) -> None:
        """Retire every provably-superseded tool result before the budget check.

        Deterministic, no-LLM pass (issue #121): replaces each retired tool
        message with a new stub message object (append-or-replace, never an
        in-place mutation) and records one passive ``supersession`` event per
        retirement. A no-op when the kill switch is set or no result
        qualifies.
        """
        if not supersession_enabled():
            return
        for retirement in find_retirements(ctx.tool_records, ctx.messages):
            artifact_path = self._artifact_path_hint(ctx, retirement.tool_call_id)
            old_message = ctx.messages[retirement.message_index]
            ctx.messages[retirement.message_index] = {
                **old_message,
                "content": build_stub(retirement, artifact_path),
            }
            ctx.recorder.record_supersession(
                tool_call_id=retirement.tool_call_id,
                tool_name=retirement.tool_name,
                case=retirement.case,
                superseding_tool_call_id=retirement.superseding_tool_call_id,
                retired_size=retirement.retired_size,
                step=ctx.step_num,
            )

    @staticmethod
    def _artifact_path_hint(ctx: RunContext, tool_call_id: str) -> str | None:
        """Return the on-disk artifact path for tool_call_id, if one exists."""
        for stream in ("stdout", "stderr"):
            path = artifact_file_path(ctx.session_id, tool_call_id, stream)
            if path is not None and path.exists():
                return str(path)
        return None

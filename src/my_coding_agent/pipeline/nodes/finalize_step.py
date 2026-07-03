"""FinalizeStepNode — record step token usage, then detect end-of-run."""

from __future__ import annotations

from ...utils import get_logger
from ...utils.parsing import extract_finish_reason, extract_usage
from ..context import RunContext
from ..node import BaseNode
from ..schema import CLEAN_FINISH_REASONS
from .context_summarizer import ContextSummarizerNode

_logger = get_logger(__name__)


class FinalizeStepNode(BaseNode):
    """Finalize a step: track token usage, then check the finish signal.

    Reads ``ctx.last_response`` once for both concerns. First it records the
    step's token usage and updates ``ctx.last_prompt_tokens``; then it sets
    ``ctx.signal`` to STOP when the model signals stop/exit/quit, else CONTINUE.
    Both the ``token_tracking`` and ``finish_check`` observability records are
    emitted so trace consumers see the same event stream as before the merge.

    This node also decides the step-ceiling cutoff explicitly: when the model
    is still going (no clean finish) on the final permitted step, it sets
    ``stop_reason = "max_steps"`` and STOP, and — when the run owes a hand-back
    to a delegating parent — triggers ``ContextSummarizerNode`` to synthesize
    the report while the conversation is still on hand.
    """

    name = "finalize_step"

    def __init__(self) -> None:
        self._summarizer = ContextSummarizerNode(kind="report", triggered_by=self.name)

    def run(self, ctx: RunContext) -> None:
        # --- token tracking ---
        usage = extract_usage(ctx.last_response)
        step_prompt = usage.get("prompt_tokens", 0)
        step_completion = usage.get("completion_tokens", 0)
        step_total = usage.get("total_tokens", 0)
        ctx.last_prompt_tokens = step_prompt
        cw = ctx.llm.context_window
        ctx_str = f" / {cw:,} ({step_prompt / cw * 100:.1f}% ctx used)" if cw else ""
        _logger.info(
            "Step %d tokens — prompt: %d, completion: %d, total: %d%s",
            ctx.step_num,
            step_prompt,
            step_completion,
            step_total,
            ctx_str,
        )
        ctx.recorder.record_token_tracking(
            step=ctx.step_num,
            prompt_tokens=step_prompt,
            completion_tokens=step_completion,
            total_tokens=step_total,
            ctx_pct=step_prompt / cw * 100 if cw else 0.0,
            context_window=cw,
        )

        # --- finish check ---
        finish_reason = extract_finish_reason(ctx.last_response)
        cutoff = False
        if finish_reason in CLEAN_FINISH_REASONS:
            ctx.stop_reason = finish_reason
            ctx.signal = "STOP"
        elif ctx.step_num >= ctx.max_steps:
            # Final permitted step with the model still going: decide the
            # cutoff here instead of leaving it to a silent loop exit.
            cutoff = True
            ctx.stop_reason = "max_steps"
            ctx.signal = "STOP"
        else:
            ctx.signal = "CONTINUE"
        ctx.recorder.record_finish_check(
            step=ctx.step_num,
            finish_reason=finish_reason,
            signal=ctx.signal,
        )
        if cutoff and ctx.needs_handback:
            self._summarizer.run(ctx)

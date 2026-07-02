"""FinalizeStepNode — record step token usage, then detect end-of-run."""

from __future__ import annotations

from ...utils import get_logger
from ...utils.parsing import extract_finish_reason, extract_usage
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class FinalizeStepNode(BaseNode):
    """Finalize a step: track token usage, then check the finish signal.

    Reads ``ctx.last_response`` once for both concerns. First it records the
    step's token usage and updates ``ctx.last_prompt_tokens``; then it sets
    ``ctx.signal`` to STOP when the model signals stop/exit/quit, else CONTINUE.
    Both the ``token_tracking`` and ``finish_check`` observability records are
    emitted so trace consumers see the same event stream as before the merge.
    """

    name = "finalize_step"

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
        if finish_reason in ("stop", "exit", "quit"):
            ctx.stop_reason = finish_reason
            ctx.signal = "STOP"
        else:
            ctx.signal = "CONTINUE"
        ctx.recorder.record_finish_check(
            step=ctx.step_num,
            finish_reason=finish_reason,
            signal=ctx.signal,
        )

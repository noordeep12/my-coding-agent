"""TokenTrackingNode — record step token usage and update the context-window counter."""

from __future__ import annotations

from ...utils import get_logger
from ...utils.parsing import extract_usage
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class TokenTrackingNode(BaseNode):
    """Read step token usage from ctx.last_response and update ctx."""

    name = "token_tracking"

    def run(self, ctx: RunContext) -> None:
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
        ctx.signal = "CONTINUE"

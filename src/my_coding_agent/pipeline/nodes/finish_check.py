"""FinishCheckNode — detect model-signalled end-of-run."""

from __future__ import annotations

from ...utils.parsing import extract_finish_reason
from ..context import RunContext
from ..node import BaseNode


class FinishCheckNode(BaseNode):
    """Set ctx.signal to STOP when the model signals stop/exit/quit."""

    name = "finish_check"

    def run(self, ctx: RunContext) -> None:
        finish_reason = extract_finish_reason(ctx.last_response)
        if finish_reason in ("stop", "exit", "quit"):
            ctx.stop_reason = finish_reason
            ctx.signal = "STOP"
        else:
            ctx.signal = "CONTINUE"

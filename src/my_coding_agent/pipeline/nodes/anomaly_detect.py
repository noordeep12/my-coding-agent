"""AnomalyDetectNode — deterministic in-run detection of failure streaks.

Runs after ``ToolDispatchNode`` and before ``FinalizeStepNode``. Reads
``ctx.tool_records`` and ``ctx.last_response`` only; writes only to the
recorder, the logger, and its own instance state. Never touches
``ctx.messages``, ``ctx.signal``, or ``ctx.stop_reason``, and makes no LLM
calls — detection-only, per the non-interference guarantee.
"""

from __future__ import annotations

from ...engine.llm import parsing as llm_parsing
from ...observability.schema import ANOMALY_KIND_FAILURE_STREAK
from ...utils import get_logger
from ..anomaly import STREAK_THRESHOLD, trailing_streak
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class AnomalyDetectNode(BaseNode):
    """Detect same-class tool-failure streaks and record them, live.

    Tracks the trailing failure streak across steps in instance state (one
    instance per pipeline, hence per session — the same pattern
    ``FinalizeStepNode`` uses for its nested summarizer). At the 3rd
    consecutive same-signature failure it assigns a ``streak_id``, logs a
    single WARNING, and records an ``anomaly`` row; each further
    same-signature failure records an updated row under the same
    ``streak_id`` with the grown length and accumulated tokens.
    """

    name = "anomaly_detect"

    def __init__(self) -> None:
        self._signature: str | None = None
        self._streak_id: str | None = None
        self._tokens_spent = 0
        self._last_length = 0
        self._ordinal = 0

    def run(self, ctx: RunContext) -> None:
        streak = trailing_streak(ctx.tool_records)
        if streak is None:
            self._signature = None
            self._streak_id = None
            self._tokens_spent = 0
            self._last_length = 0
            return

        signature, length, indexes = streak
        if signature != self._signature:
            self._signature = signature
            self._streak_id = None
            self._tokens_spent = 0
            self._last_length = 0

        if length <= self._last_length:
            # No new same-signature failure landed this step (e.g. the model
            # finished cleanly without another tool call) — nothing grew.
            return
        self._last_length = length

        usage = llm_parsing.extract_usage(ctx.last_response)
        self._tokens_spent += usage.get("prompt_tokens", 0) + usage.get(
            "completion_tokens", 0
        )

        if length < STREAK_THRESHOLD:
            return

        tool_name = ctx.tool_records[indexes[-1]].get("name", "")
        first_signal = self._streak_id is None
        if first_signal:
            self._ordinal += 1
            self._streak_id = f"{ctx.session_id[:8]}-{self._ordinal}"
            _logger.warning(
                "Anomaly: %s failed %d times in a row (signature %s, %d tokens spent)",
                tool_name,
                length,
                signature,
                self._tokens_spent,
            )

        ctx.recorder.record_anomaly(
            kind=ANOMALY_KIND_FAILURE_STREAK,
            streak_id=self._streak_id or "",
            signature=signature,
            tool_name=tool_name,
            streak_len=length,
            tokens_spent=self._tokens_spent,
            step=ctx.step_num,
        )

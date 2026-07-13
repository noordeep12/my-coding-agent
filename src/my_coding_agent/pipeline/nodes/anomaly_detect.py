"""AnomalyDetectNode — deterministic in-run detection of failure streaks.

Runs after ``ToolDispatchNode`` and before ``FinalizeStepNode``. Reads
``ctx.tool_records`` and ``ctx.last_response`` only; writes only to the
recorder, the logger, and its own instance state. Never touches
``ctx.messages``, ``ctx.signal``, or ``ctx.stop_reason``, and makes no LLM
calls — detection-only, per the non-interference guarantee.
"""

from __future__ import annotations

from typing import Any

from ...engine.llm import parsing as llm_parsing
from ...observability import classify_error
from ...observability.schema import ANOMALY_KIND_FAILURE_STREAK
from ...utils import get_logger
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)

# Threshold at which a same-signature failure streak is signaled. Not
# configurable, per simplicity-first (confirmed by the motivating metric).
STREAK_THRESHOLD = 3


def error_signature(record: dict[str, Any]) -> str:
    """Return ``"<tool_name>|<error_class>"`` for a failed tool record.

    ``error_class`` comes from the shared classification helper (same rule
    the recorder uses for ``tool_call`` events' ``error_class``), so grouping
    by signature here and by ``error_class`` there agree by construction. Args
    never participate in the signature.

    Args:
        record: A tool-call record as appended to ``ctx.tool_records`` (must
            have ``name`` and ``error`` keys).

    Returns:
        The signature string identifying this failure's class.
    """
    tool_name = record.get("name", "")
    error_text = str(record.get("error", ""))
    return f"{tool_name}|{classify_error(error_text)}"


def trailing_streak(
    tool_records: list[dict[str, Any]],
) -> tuple[str, int, list[int]] | None:
    """Return the current trailing same-signature failure streak, if any.

    Scans ``tool_records`` from the end backwards: a success or a signature
    change stops the scan. Returns ``None`` when the trailing record is a
    success or there are no records.

    Args:
        tool_records: The run's accumulated tool-call records in call order.

    Returns:
        A ``(signature, length, member_indexes)`` tuple for the trailing
        streak (indexes ascending, into ``tool_records``), or ``None`` if the
        run does not currently end on a failure.
    """
    if not tool_records:
        return None
    last = tool_records[-1]
    if last.get("ok", True):
        return None
    signature = error_signature(last)
    member_indexes: list[int] = []
    for idx in range(len(tool_records) - 1, -1, -1):
        record = tool_records[idx]
        if record.get("ok", True):
            break
        if error_signature(record) != signature:
            break
        member_indexes.append(idx)
    member_indexes.reverse()
    return signature, len(member_indexes), member_indexes


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

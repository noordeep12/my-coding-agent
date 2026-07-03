"""Pure helpers for failure-streak detection — signature and streak scan.

Kept outside ``nodes/`` (per the one-``BaseNode``-per-module rule) so the
signature extraction and streak-scan logic are unit-testable in isolation
from the pipeline node that consumes them (``nodes/anomaly_detect.py``).
"""

from __future__ import annotations

import re
from typing import Any

# Threshold at which a same-signature failure streak is signaled. Not
# configurable, per simplicity-first (confirmed by the motivating metric).
STREAK_THRESHOLD = 3

# Last Python-exception-style token in an error string, e.g. the
# ``json.decoder.JSONDecodeError`` in a bash traceback. Matches a dotted
# identifier ending in ``Error`` or ``Exception``.
_EXC_TOKEN_RE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception)\b")

# Digits stripped from the fallback bucket so wording that differs only by a
# number (line numbers, counts) still buckets together.
_DIGITS_RE = re.compile(r"\d+")

_FALLBACK_MAX_LEN = 80


def error_signature(record: dict[str, Any]) -> str:
    """Return ``"<tool_name>|<error_class>"`` for a failed tool record.

    ``error_class`` is the last ``…Error``/``…Exception``-style token found in
    the record's ``error`` text when one is present (e.g.
    ``json.decoder.JSONDecodeError`` inside a bash traceback); otherwise it
    falls back to the error text's first line, digits stripped and truncated,
    as a crude bucket. Args never participate in the signature.

    Args:
        record: A tool-call record as appended to ``ctx.tool_records`` (must
            have ``name`` and ``error`` keys).

    Returns:
        The signature string identifying this failure's class.
    """
    tool_name = record.get("name", "")
    error_text = str(record.get("error", ""))
    matches = _EXC_TOKEN_RE.findall(error_text)
    if matches:
        error_class = matches[-1]
    else:
        first_line = error_text.splitlines()[0] if error_text else ""
        normalized = _DIGITS_RE.sub("", first_line).strip()
        error_class = normalized[:_FALLBACK_MAX_LEN]
    return f"{tool_name}|{error_class}"


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

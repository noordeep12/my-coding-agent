"""Pure helpers for deterministic tool-result supersession — Cases A, B, C.

Kept outside ``nodes/`` (per the one-``BaseNode``-per-module rule) so the
identification rules and stub text are unit-testable in isolation from the
pipeline node that applies them (``nodes/context_guard.py``), mirroring the
``anomaly.py`` / ``nodes/anomaly_detect.py`` split. Retirement only ever
*replaces* a tool message's content with a one-line stub — never deletes a
message or mutates one in place — so the recorder's append-or-replace
invariant holds and trace/checkpoint fidelity is unaffected.

See ``openspec/changes/tool-result-supersession/design.md`` for the three
provable cases and why heuristic similarity is out of scope.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, NamedTuple

from ..engine.tool_execution.schema import (
    EXTRACTION_INCOMPLETE_MARKER,
    SUPERSESSION_SIZE_FLOOR_CHARS,
)

CASE_INCOMPLETE_EXTRACT = "incomplete_extract"  # Case A
CASE_CONTAINMENT = "containment"  # Case B
CASE_IDENTICAL_CALL = "identical_call"  # Case C

# Prefix marking a message this pass already retired, so a later step's pass
# treats an already-stubbed message as inert rather than retiring it again.
STUB_PREFIX = "[Superseded —"

# Kill switch: set MCA_SUPERSESSION=0 to disable the pass entirely and restore
# append-only behavior byte-for-byte (mirrors the MCA_TOOL_MAX_CONCURRENCY
# precedent in engine/tool_execution/concurrency.py).
_SUPERSESSION_ENV = "MCA_SUPERSESSION"


class Retirement(NamedTuple):
    """One tool message to retire: which message, why, and by what."""

    message_index: int
    tool_call_id: str
    tool_name: str
    case: str
    superseding_tool_call_id: str
    retired_size: int


def supersession_enabled() -> bool:
    """Return whether the supersession pass runs, read from ``MCA_SUPERSESSION``.

    Read at call time (not import time), matching the
    ``max_tool_concurrency`` precedent. Enabled by default; ``"0"`` disables
    it and restores today's append-only behavior byte-for-byte.
    """
    return os.environ.get(_SUPERSESSION_ENV, "1") != "0"


def _tool_message_index(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Map each tool-role message's ``tool_call_id`` to its position."""
    index: dict[str, int] = {}
    for i, message in enumerate(messages):
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str):
                index[call_id] = i
    return index


def _content_text(messages: list[dict[str, Any]], message_index: int | None) -> str:
    """Return the current message content string at ``message_index``, or ''."""
    if message_index is None:
        return ""
    content = messages[message_index].get("content")
    return content if isinstance(content, str) else ""


def _args_signature(args: Any) -> str:
    """Deterministic signature for a call's arguments (key order independent)."""
    return json.dumps(args, sort_keys=True, default=str)


# A case finder calls this with (tool_call_id, tool_name, superseding_tool_call_id)
# for each candidate; the case itself is already bound into the callback.
_Consider = Callable[[str, str, str], None]


def _find_identical_call_retirements(
    tool_records: list[dict[str, Any]], consider: _Consider
) -> None:
    """Case C — retire every older invocation of a byte-identical ``(name,
    args)`` call once its newest invocation succeeded."""
    by_signature: dict[str, list[dict[str, Any]]] = {}
    for record in tool_records:
        sig = f"{record.get('name')}|{_args_signature(record.get('args'))}"
        by_signature.setdefault(sig, []).append(record)
    for records in by_signature.values():
        if len(records) < 2:
            continue
        newest = records[-1]
        if not newest.get("ok"):
            continue
        newest_id = newest.get("tool_call_id", "")
        for older in records[:-1]:
            call_id = older.get("tool_call_id", "")
            if call_id != newest_id:
                consider(call_id, older.get("name", ""), newest_id)


def _find_incomplete_extract_retirements(
    tool_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    index: dict[str, int],
    consider: _Consider,
) -> None:
    """Case A — retire a marked-incomplete extract of artifact X once a later
    successful call reads that same artifact X."""
    for i, record in enumerate(tool_records):
        if record.get("name") != "read_tool_artifact":
            continue
        call_id = record.get("tool_call_id", "")
        text = _content_text(messages, index.get(call_id))
        if EXTRACTION_INCOMPLETE_MARKER not in text:
            continue
        target = (record.get("args") or {}).get("tool_call_id")
        if not target:
            continue
        for later in tool_records[i + 1 :]:
            later_target = (later.get("args") or {}).get("tool_call_id")
            if later.get("ok") and later_target == target:
                consider(call_id, record.get("name", ""), later.get("tool_call_id", ""))
                break


def _find_containment_retirements(
    tool_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    index: dict[str, int],
    consider: _Consider,
) -> None:
    """Case B — retire an earlier successful result whose full text is a
    contiguous substring of a later successful result's text."""
    successes = [r for r in tool_records if r.get("ok")]
    for i, earlier in enumerate(successes):
        earlier_id = earlier.get("tool_call_id", "")
        earlier_text = _content_text(messages, index.get(earlier_id))
        if not earlier_text:
            continue
        for later in successes[i + 1 :]:
            later_id = later.get("tool_call_id", "")
            later_text = _content_text(messages, index.get(later_id))
            if len(later_text) >= len(earlier_text) and earlier_text in later_text:
                consider(earlier_id, earlier.get("name", ""), later_id)
                break


def find_retirements(
    tool_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    size_floor: int = SUPERSESSION_SIZE_FLOOR_CHARS,
) -> list[Retirement]:
    """Identify every tool message provably superseded by a later one.

    Applies, in order, Case C (an older invocation of a byte-identical
    ``(name, args)`` call, superseded by its newest successful invocation),
    Case A (a ``read_tool_artifact`` extract carrying the incompleteness
    marker for artifact X, superseded once a later successful call reads that
    same artifact X), and Case B (an earlier successful result's full text is
    a contiguous substring of a later successful result's text). A message
    already holding the retirement stub is left alone, so repeated passes
    across steps are idempotent. Below ``size_floor``, a result is never
    retired regardless of case.

    Args:
        tool_records: The run's accumulated tool-call records, in call order.
        messages: The live conversation, in order.
        size_floor: Minimum retired-content size, in characters.

    Returns:
        Retirements to apply, ordered by ``message_index``. Each
        ``message_index`` appears at most once.
    """
    index = _tool_message_index(messages)
    retirements: dict[int, Retirement] = {}

    def _consider_for(case: str) -> _Consider:
        def consider(tool_call_id: str, tool_name: str, superseding_id: str) -> None:
            msg_index = index.get(tool_call_id)
            if msg_index is None or msg_index in retirements:
                return
            text = _content_text(messages, msg_index)
            if text.startswith(STUB_PREFIX) or len(text) < size_floor:
                return
            retirements[msg_index] = Retirement(
                msg_index, tool_call_id, tool_name, case, superseding_id, len(text)
            )

        return consider

    _find_identical_call_retirements(tool_records, _consider_for(CASE_IDENTICAL_CALL))
    _find_incomplete_extract_retirements(
        tool_records, messages, index, _consider_for(CASE_INCOMPLETE_EXTRACT)
    )
    _find_containment_retirements(
        tool_records, messages, index, _consider_for(CASE_CONTAINMENT)
    )

    return sorted(retirements.values(), key=lambda r: r.message_index)


def build_stub(retirement: Retirement, artifact_path: str | None) -> str:
    """Build the one-line, recoverable stub for a retired tool message.

    Names the producing tool, its ``tool_call_id``, the superseding call, and
    — when the result had an on-disk artifact — the artifact path. The
    original content is never deleted: it remains reachable via
    ``read_tool_artifact``, the on-disk file, or the run's trace (whose
    recorder falls back to a full snapshot on a retiring step).
    """
    where = f" On disk at {artifact_path}." if artifact_path else ""
    return (
        f"{STUB_PREFIX} {retirement.tool_name} result "
        f"(tool_call_id={retirement.tool_call_id!r}) retired as "
        f"{retirement.case}, superseded by tool_call_id="
        f"{retirement.superseding_tool_call_id!r}.{where} Recover via "
        f"read_tool_artifact(tool_call_id={retirement.tool_call_id!r}) "
        "or the run trace."
    )

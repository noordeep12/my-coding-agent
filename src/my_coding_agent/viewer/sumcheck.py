"""Deterministic, LLM-free sum-check over a session directory tree (D4).

Verifies, without any model call: the sum of a session's own per-call usage
rows per kind equals its persisted rollup ``by_kind``; its own totals plus its
descendants' grand totals equal its ``grand_total``; and a report event's
provenance is consistent with the presence or absence of a ``report``-kind
usage row (D2). Incomplete records (a crash before ``session_data.json`` was
written, or a pre-provenance report event) are reported as unverifiable, never
as a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass
class SessionCheck:
    """The verdict for one session directory: pass, fail, or unverifiable."""

    session_id: str
    status: str  # "pass" | "fail" | "unverifiable"
    reasons: list[str] = field(default_factory=list)


def _sum_calls_by_kind(llm_calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_kind: dict[str, dict[str, int]] = {}
    for call in llm_calls:
        agg = by_kind.setdefault(
            call.get("kind", "main"), dict.fromkeys(_TOKEN_KEYS, 0)
        )
        agg["prompt_tokens"] += call.get("prompt", 0)
        agg["completion_tokens"] += call.get("completion", 0)
        agg["total_tokens"] += call.get("total", 0)
    return by_kind


def _sum_totals(by_kind: dict[str, dict[str, int]]) -> dict[str, int]:
    total = dict.fromkeys(_TOKEN_KEYS, 0)
    for agg in by_kind.values():
        for key in _TOKEN_KEYS:
            total[key] += agg[key]
    return total


_NO_REPORT = "__no_report__"
_MISSING_SOURCE = "__missing_source__"


def _report_source(events_path: Path) -> str:
    """Return the session's report event source, or a sentinel: no report
    event exists (``_NO_REPORT``), or one exists without a ``source`` key —
    a pre-provenance trace (``_MISSING_SOURCE``).
    """
    if not events_path.exists():
        return _NO_REPORT
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("type") == "report":
            return str(event.get("source", _MISSING_SOURCE))
    return _NO_REPORT


def check_session(session_dir: Path) -> SessionCheck:
    """Check one session directory's own arithmetic and report provenance.

    Does not recurse into descendants — see ``check_tree`` for the whole tree.
    """
    session_id = session_dir.name
    data_path = session_dir / "session_data.json"
    if not data_path.exists():
        return SessionCheck(session_id, "unverifiable", ["missing session_data.json"])

    data = json.loads(data_path.read_text(encoding="utf-8"))
    rollup = data.get("rollup") or {}
    reasons: list[str] = []
    status = "pass"

    # (a) sum of own llm_calls rows per kind == rollup's by_kind
    own_calls = data.get("llm_calls") or []
    computed_by_kind = _sum_calls_by_kind(own_calls)
    persisted_by_kind = rollup.get("by_kind") or {}
    if computed_by_kind != persisted_by_kind:
        status = "fail"
        kinds = sorted(set(computed_by_kind) | set(persisted_by_kind))
        reasons.append(f"by_kind mismatch for kind(s): {', '.join(kinds)}")

    # (b) own totals + descendants' grand totals == grand_total
    own_total = _sum_totals(computed_by_kind)
    descendants = rollup.get("descendants") or []
    expected_grand = dict(own_total)
    for child in descendants:
        child_total = child.get("grand_total") or {}
        for key in _TOKEN_KEYS:
            expected_grand[key] += child_total.get(key, 0)
    persisted_grand = rollup.get("grand_total") or {}
    if expected_grand != persisted_grand:
        status = "fail"
        reasons.append("grand_total does not equal own totals plus descendants")

    # (c) report provenance is consistent with the presence/absence of a
    # report-kind usage row (D2)
    source = _report_source(session_dir / "events.jsonl")
    report_kind_rows = sum(1 for call in own_calls if call.get("kind") == "report")
    if source == _MISSING_SOURCE:
        if status == "pass":
            status = "unverifiable"
        reasons.append("report event predates provenance (no source)")
    elif source == "verbatim" and report_kind_rows != 0:
        status = "fail"
        reasons.append("verbatim report but a report-kind usage row exists")
    elif source in ("summarizer", "fallback") and report_kind_rows != 1:
        status = "fail"
        reasons.append(
            f"{source} report but {report_kind_rows} report-kind usage row(s) "
            "exist (expected 1)"
        )

    return SessionCheck(session_id, status, reasons)


def check_tree(base_dir: Path, session_id: str) -> list[SessionCheck]:
    """Check a session and every descendant named in its rollup.

    Descendant session ids are read from ``rollup["descendants"]`` and
    resolved as sibling directories under ``base_dir`` (D3's own directory
    tree), so the whole delegated subtree is verified without an LLM call.
    """
    session_dir = base_dir / session_id
    results = [check_session(session_dir)]
    data_path = session_dir / "session_data.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        for child in (data.get("rollup") or {}).get("descendants") or []:
            child_id = child.get("session_id")
            if child_id:
                results.extend(check_tree(base_dir, child_id))
    return results

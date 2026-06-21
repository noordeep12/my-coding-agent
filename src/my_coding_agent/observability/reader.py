"""Read and derive views from a session's event stream.

This is the shared model layer the HTML viewer (via ``tree``/``report``) sits on.
``load_session`` parses ``events.jsonl`` (joining
in ``ok``/``status`` from the untouched ``session_data.json``) into a typed
``Session``; the module-level functions derive the analytical views the viewer
renders: context-growth series, message diffs, cost/latency bottlenecks,
loop/duplicate detection, and extracted code blocks.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from . import events, pricing
from .events import LLMCall, Session, ToolCall

DEFAULT_ROOT = ".my_coding_agent"


# ── loading ────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of dicts, skipping blank/corrupt lines."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a half-written final line from a crashed run
    return rows


def load_session(session_dir: str | Path) -> Session | None:
    """Load one session directory into a ``Session``, or ``None`` if uninstrumented.

    Args:
        session_dir: A ``.my_coding_agent/<session_id>/`` directory.

    Returns:
        The parsed ``Session``, or ``None`` when no ``events.jsonl`` is present
        (e.g. a run that predates the observability layer).
    """
    session_dir = Path(session_dir)
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return None
    rows = _read_jsonl(events_path)

    start = next((r for r in rows if r.get("type") == events.SESSION_START), {})
    end = next((r for r in rows if r.get("type") == events.SESSION_END), {})
    session = Session(
        session_id=str(start.get("session_id", session_dir.name)),
        label=str(start.get("label", "Agent")),
        model=str(start.get("model", "")),
        parent_session_id=start.get("parent_session_id"),
        context_window=int(start.get("context_window", 0)),
        started_at=str(start.get("started_at", "")),
        stop_reason=str(end.get("stop_reason", "unknown")),
        steps=int(end.get("steps", 0)),
        elapsed_s=float(end.get("elapsed_s", 0.0)),
    )

    for r in rows:
        kind = r.get("type")
        if kind == events.LLM_CALL:
            session.llm_calls.append(LLMCall.from_dict(r))
            session.timeline.append(r)
        elif kind == events.TOOL_CALL:
            session.tool_calls.append(ToolCall.from_dict(r))
            session.timeline.append(r)
        elif kind == events.HANDOFF:
            session.handoffs.append(events.Handoff.from_dict(r))
            session.timeline.append(r)
        elif kind == events.ROUTER:
            session.timeline.append(r)

    _join_tool_status(session, session_dir / "session_data.json")
    return session


def _join_tool_status(session: Session, session_data_path: Path) -> None:
    """Backfill ok/status/tool_call_id onto tool calls from ``session_data.json``.

    The before/after hooks do not see the tool_call_id or success flag; both lists
    are ordered identically, so a positional zip recovers them.
    """
    if not session_data_path.exists():
        return
    try:
        data = json.loads(session_data_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    records = data.get("tool_records", []) or []
    for call, record in zip(session.tool_calls, records):
        call.ok = record.get("ok")
        call.status = record.get("status")
        call.tool_call_id = record.get("tool_call_id")


def load_sessions_by_id(root: str | Path = DEFAULT_ROOT) -> dict[str, Session]:
    """Load every instrumented session under ``root`` into a flat ``{id: Session}``.

    Used by the trace-tree viewer, which nests delegated subagents itself via
    ``child_session_id`` rather than the pre-nested ``children`` lists.
    """
    root = Path(root)
    out: dict[str, Session] = {}
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if child.is_dir():
            session = load_session(child)
            if session is not None:
                out[session.session_id] = session
    return out


def load_all_sessions(root: str | Path = DEFAULT_ROOT) -> list[Session]:
    """Load every instrumented session under ``root``, nesting children by parent.

    Returns:
        Top-level sessions (``parent_session_id is None`` or an absent parent),
        each with its delegated/continuation children attached under ``children``.
    """
    root = Path(root)
    if not root.exists():
        return []
    by_id: dict[str, Session] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        session = load_session(child)
        if session is not None:
            by_id[session.session_id] = session

    roots: list[Session] = []
    for session in by_id.values():
        parent = by_id.get(session.parent_session_id or "")
        if parent is not None and parent is not session:
            parent.children.append(session)
        else:
            roots.append(session)
    roots.sort(key=lambda s: s.started_at, reverse=True)
    return roots


# ── derived views ──────────────────────────────────────────────────────────────
def context_series(session: Session) -> dict[str, list[Any]]:
    """Per-call context-window usage for the growth chart.

    Returns:
        Columns ``call``, ``prompt_tokens``, ``context_window``, ``pct`` over the
        main/handoff calls (the ones that carry the conversation).
    """
    calls = [c for c in session.llm_calls if c.prompt > 0]
    return {
        "call": [c.call for c in calls],
        "prompt_tokens": [c.prompt for c in calls],
        "context_window": [c.context_window for c in calls],
        "pct": [round(c.context_pct, 1) for c in calls],
    }


def _msg_signature(msg: dict[str, Any]) -> str:
    """A compact, stable signature for one message used in diffing."""
    role = msg.get("role", "?")
    content = msg.get("content") or ""
    if not content and msg.get("tool_calls"):
        names = ",".join(
            tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]
        )
        content = f"<tool_calls: {names}>"
    return f"{role}: {content[:200]}"


def message_diff(prev: LLMCall, curr: LLMCall) -> list[tuple[str, str]]:
    """Diff the conversation between two consecutive snapshot-bearing calls.

    Returns:
        ``(tag, line)`` pairs where ``tag`` is ``"+"`` (added), ``"-"`` (removed),
        or ``" "`` (unchanged), built from per-message signatures. Empty when
        either call lacks a stored snapshot.
    """
    if not prev.messages or not curr.messages:
        return []
    a = [_msg_signature(m) for m in prev.messages]
    b = [_msg_signature(m) for m in curr.messages]
    diff: list[tuple[str, str]] = []
    for line in difflib.ndiff(a, b):
        tag = line[:1]
        if tag in ("+", "-", " "):
            diff.append((tag, line[2:]))
    return diff


def bottlenecks(session: Session) -> list[dict[str, Any]]:
    """Per-tool and per-LLM-kind aggregates, sortable by tokens/latency/cost.

    Returns:
        Rows with ``step`` (tool name or ``llm:<kind>``), ``calls``, ``tokens``,
        ``latency_s``, and estimated ``cost_usd`` — the bottleneck table.
    """
    rows: dict[str, dict[str, Any]] = {}

    def bucket(key: str) -> dict[str, Any]:
        return rows.setdefault(
            key,
            {"step": key, "calls": 0, "tokens": 0, "latency_s": 0.0, "cost_usd": 0.0},
        )

    for c in session.llm_calls:
        row = bucket(f"llm:{c.kind}")
        row["calls"] += 1
        row["tokens"] += c.total
        row["latency_s"] = round(row["latency_s"] + c.latency_s, 4)
        row["cost_usd"] = round(
            row["cost_usd"] + pricing.cost_usd(session.model, c.prompt, c.completion), 6
        )
    for t in session.tool_calls:
        row = bucket(t.name)
        row["calls"] += 1
        row["latency_s"] = round(row["latency_s"] + t.latency_s, 4)
    return sorted(rows.values(), key=lambda r: r["tokens"], reverse=True)


def detect_loops(session: Session) -> list[dict[str, Any]]:
    """Flag repeated identical tool calls and repeated assistant decisions.

    Returns:
        Rows with ``kind`` (``"tool"``/``"decision"``), ``signature``, and
        ``count`` for any identical call/decision seen more than once.
    """
    flags: list[dict[str, Any]] = []

    tool_seen: dict[str, int] = {}
    for t in session.tool_calls:
        sig = f"{t.name}({json.dumps(t.args, sort_keys=True, default=str)})"
        tool_seen[sig] = tool_seen.get(sig, 0) + 1
    flags += [
        {"kind": "tool", "signature": sig, "count": n}
        for sig, n in tool_seen.items()
        if n > 1
    ]

    decision_seen: dict[str, int] = {}
    for c in session.llm_calls:
        content = (c.response.get("content") or "").strip()
        if content:
            decision_seen[content] = decision_seen.get(content, 0) + 1
    flags += [
        {"kind": "decision", "signature": sig[:120], "count": n}
        for sig, n in decision_seen.items()
        if n > 1
    ]
    return sorted(flags, key=lambda f: f["count"], reverse=True)


def extract_code_blocks(session: Session) -> list[dict[str, str]]:
    """Pull executed/generated code into formatted, readable blocks.

    Sources: ``write_file`` content, ``bash`` commands, and fenced blocks in
    assistant message content.

    Returns:
        Rows with ``source``, ``language``, and ``code``.
    """
    blocks: list[dict[str, str]] = []
    for t in session.tool_calls:
        if t.name == "write_file" and t.args.get("content"):
            blocks.append(
                {
                    "source": f"write_file {t.args.get('file_path', '')}",
                    "language": _guess_language(str(t.args.get("file_path", ""))),
                    "code": str(t.args["content"]),
                }
            )
        elif t.name == "bash" and t.args.get("command"):
            blocks.append(
                {"source": "bash", "language": "bash", "code": str(t.args["command"])}
            )
    for c in session.llm_calls:
        for lang, code in _fenced_blocks(c.response.get("content") or ""):
            blocks.append(
                {
                    "source": f"assistant (call #{c.call})",
                    "language": lang,
                    "code": code,
                }
            )
    return blocks


_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".sh": "bash",
    ".md": "markdown",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _guess_language(file_path: str) -> str:
    """Map a file extension to a syntax-highlight language ("text" fallback)."""
    return _EXT_LANG.get(Path(file_path).suffix, "text")


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Extract ``(language, code)`` pairs from triple-backtick fences in text."""
    blocks: list[tuple[str, str]] = []
    parts = text.split("```")
    for i in range(1, len(parts), 2):  # odd segments are inside fences
        body = parts[i]
        head, _, rest = body.partition("\n")
        lang = head.strip() or "text"
        blocks.append((lang, rest if rest else head))
    return blocks

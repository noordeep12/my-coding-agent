"""Call-record builders for the agent's tool history.

A *call record* is the per-tool-call dict the executor appends to
``Agent.tool_records`` and persists in ``session_data.json`` — the observability
view of what each tool did (``name``/``args``/``ok``/``status``/…). These builders
own that record shape so it stays consistent across success, failure, skip, and
parse-error paths.
"""

from typing import Any


def error_record(
    func_name: str,
    args: dict,
    tool_call_id: str,
    error: str,
    status: str = "error",
) -> dict[str, Any]:
    """Build a failure call-record (for session_data.json / observability)."""
    return {
        "name": func_name,
        "args": args,
        "ok": False,
        "error": error,
        "tool_call_id": tool_call_id,
        "artifact": False,
        "truncated": False,
        "status": status,
    }


def call_record(
    func_name: str,
    args: dict,
    tool_call_id: str,
    env: dict[str, Any],
    is_artifact: bool,
    is_truncated: bool,
) -> tuple[str, dict[str, Any]]:
    """Build the (status, call-record) for a dispatched tool, from its envelope."""
    status = "success" if env["ok"] else "error"
    record = {
        "name": func_name,
        "args": args,
        "ok": env["ok"],
        "tool_call_id": tool_call_id,
        "artifact": is_artifact,
        "truncated": is_truncated,
        "status": status,
    }
    if not env["ok"]:
        record["error"] = env["error"] or "tool reported failure"
    return status, record

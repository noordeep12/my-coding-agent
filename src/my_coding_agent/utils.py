import json
from typing import Any


def extract_message(resp) -> dict:
    """Return the message dict from a chat completion httpx response. Always returns a dict."""
    choices = resp.json().get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def extract_finish_reason(resp) -> str:
    """Return the lowercased finish_reason from a chat completion httpx response."""
    choices = resp.json().get("choices") or []
    reason = choices[0].get("finish_reason") if choices else None
    return (reason or "").lower()


def extract_usage(resp) -> dict:
    """Return token usage dict from a chat completion httpx response. Always returns a dict."""
    return resp.json().get("usage") or {}


def parse_tool_args(args: Any) -> dict:
    """Coerce tool call arguments to a dict (handles both str-JSON and dict)."""
    return json.loads(args) if isinstance(args, str) else args

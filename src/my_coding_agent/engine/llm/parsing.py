"""Thin chat-completion response-parsing helpers.

Small pure functions that extract fields from a chat-completion ``httpx``
response (message, finish reason, token usage) and parse tool-call arguments,
keeping that boilerplate out of the LLM client, routing, tool execution, and
pipeline nodes that consume the OpenAI-compatible wire format.
"""

import json
from typing import Any

from httpx import Response


def extract_message(resp: Response) -> dict[str, Any]:
    """Return the message dict from a chat completion response (always a dict)."""
    choices = resp.json().get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def extract_finish_reason(resp: Response) -> str:
    """Return the lowercased finish_reason from a chat completion httpx response."""
    choices = resp.json().get("choices") or []
    reason = choices[0].get("finish_reason") if choices else None
    return (reason or "").lower()


def extract_usage(resp: Response) -> dict[str, Any]:
    """Return token usage dict from a chat completion response (always a dict)."""
    return resp.json().get("usage") or {}


def parse_tool_args(args: Any) -> dict[str, Any]:
    """Coerce tool call arguments to a dict (handles both str-JSON and dict)."""
    result: dict[str, Any] = json.loads(args) if isinstance(args, str) else args
    return result

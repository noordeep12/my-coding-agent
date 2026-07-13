"""Shared agent-run helpers for a config-driven eval run (see run_config_file.py)."""

from __future__ import annotations

import inspect
from typing import Any

from ..engine.tool_registry import ToolRegistry, tool


def _build_tools() -> list[dict[str, Any]]:
    names = [
        name
        for name, _ in inspect.getmembers(ToolRegistry, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    return [tool(getattr(ToolRegistry, name)) for name in names]


def _final_output(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""

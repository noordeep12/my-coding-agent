"""Language-hint resolver for the tool-result envelope.

Produces the advisory ``metadata.lang = {command, output, error}`` hint from the
tool name and call args, so a consumer (the trace viewer) renders each box in its
language without re-inferring it. The value set is closed — ``python | shell |
json | text`` — and unknown content falls back to ``text``.

This module is pure data logic: no LLM client, no execution state, no I/O. The
``ToolExecutor`` composes it, annotating each envelope before serialization.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Literal

Lang = Literal["python", "shell", "json", "text"]

# File extension → language for a ``read_file`` output. Minimal by design
# (extend on demand); anything not listed falls back to ``text``.
_EXT_LANG: dict[str, Lang] = {
    ".py": "python",
    ".pyi": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".json": "json",
}


def _looks_json(text: str) -> bool:
    """Return True when ``text`` parses as a JSON object or array."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


def _command_lang(tool: str, args: Any) -> Lang:
    """Language of the invocation box: ``bash`` is a shell command, else args JSON."""
    if tool == "bash":
        return "shell"
    return "json" if isinstance(args, dict) else "text"


def _output_lang(tool: str, args: Any, output: str) -> Lang:
    """Language of the output box: file extension for reads, else a JSON sniff."""
    if tool == "read_file" and isinstance(args, dict):
        file_path = args.get("file_path")
        if isinstance(file_path, str):
            return _EXT_LANG.get(PurePosixPath(file_path).suffix.lower(), "text")
    if _looks_json(output):
        return "json"
    return "text"


def resolve_lang(tool: str, args: Any, env: dict[str, Any]) -> dict[str, Lang]:
    """Resolve the ``{command, output, error}`` language hint for one envelope."""
    return {
        "command": _command_lang(tool, args),
        "output": _output_lang(tool, args, env.get("output", "")),
        "error": "text",
    }

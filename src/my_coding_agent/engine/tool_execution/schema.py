"""Canonical tool-output schema — the agent-facing result contract.

Every tool result that reaches the agent — including auto-triggered paths
(artifact summaries, skips, parse/arg errors, raised exceptions) — is normalized
into one envelope so success/failure is uniform and machine-checkable, modeled on
bash's ``ok``/``exit_code``. ``output`` carries the raw payload (stdout / file
content / report / summary); tool-specific extras go in the flexible ``metadata``
bag.

This module is pure data logic: no LLM client, no execution state, no I/O. The
``ToolExecutor`` composes it, injecting the stored artifact into
:func:`result_envelope`.
"""

import json
import re
from typing import Any

TOOL_SCHEMA_VERSION = 1
_TOOL_RESULT_KEYS = ("schema_version", "tool", "ok", "output", "error", "metadata")
_ERROR_PREFIX_RE = re.compile(r"^Error\b")

# Shape of ``metadata.preview`` for an offloaded artifact: a descriptor of the
# bounded excerpt carried in ``output``. The excerpt itself and the skim guidance
# live in ``output`` (not here); the full raw output lives only on disk.
PREVIEW_KEYS = (
    "shown_lines",
    "total_lines",
    "shown_bytes",
    "total_bytes",
    "full_output_path",
)


def build_tool_result(
    tool: str,
    ok: bool,
    output: str = "",
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical tool-result envelope (the agent-facing contract)."""
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "tool": tool,
        "ok": bool(ok),
        "output": output if isinstance(output, str) else str(output),
        "error": error,
        "metadata": metadata or {},
    }


def validate_tool_result(result: Any) -> dict[str, Any]:
    """Enforce the schema; raise ``ValueError`` if a result does not conform."""
    if not isinstance(result, dict):
        raise ValueError(f"tool result must be a dict, got {type(result).__name__}")
    missing = [k for k in _TOOL_RESULT_KEYS if k not in result]
    if missing:
        raise ValueError(f"tool result missing keys: {missing}")
    if not isinstance(result["ok"], bool):
        raise ValueError("tool result 'ok' must be a bool")
    if not isinstance(result["output"], str):
        raise ValueError("tool result 'output' must be a str")
    if result["error"] is not None and not isinstance(result["error"], str):
        raise ValueError("tool result 'error' must be a str or None")
    if not isinstance(result["metadata"], dict):
        raise ValueError("tool result 'metadata' must be a dict")
    return result


def _maybe_json(text: Any) -> Any:
    """Parse ``text`` as JSON, returning None when it is not JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _structured_envelope(
    tool: str, data: dict[str, Any], output: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build an envelope from a bash-style ``{ok, exit_code, stderr}`` result."""
    ok = bool(data.get("ok", True))
    for key in ("exit_code", "stderr"):
        if key in data:
            metadata[key] = data[key]
    error = None if ok else (data.get("stderr") or "command failed")
    return build_tool_result(tool, ok, output, error, metadata)


def result_envelope(
    tool: str,
    result: str,
    is_artifact: bool,
    is_truncated: bool,
    tool_call_id: str,
    artifact: Any = None,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a tool's raw return into the canonical schema envelope.

    Detects failure via structured returns (bash-family ``ok``/``exit_code``, or
    the stored ``artifact`` for an offloaded bash output) and the ``Error…`` string
    convention the file/web tools use, defaulting to success otherwise.

    Args:
        tool: Tool name.
        result: The agent-facing string the tool produced. For an offloaded
            artifact this is the bounded preview excerpt plus inline skim guidance.
        is_artifact: Whether ``result`` is the preview of an offloaded artifact.
        is_truncated: Whether ``result`` was truncated to the output limit.
        tool_call_id: The call id, echoed into artifact metadata.
        artifact: The stored full artifact for this call, if any (injected by the
            executor so this function stays pure).
        preview: The ``preview`` descriptor (see ``PREVIEW_KEYS``) for an offloaded
            artifact. When supplied it is attached to ``metadata.preview``.
    """
    metadata: dict[str, Any] = {}
    if is_truncated:
        metadata["truncated"] = True

    # bash-family: structured JSON carrying its own ok/exit_code.
    parsed = _maybe_json(result)
    if isinstance(parsed, dict) and "ok" in parsed:
        output = parsed.get("stdout", "")
        return _structured_envelope(tool, parsed, output, metadata)

    # bash large output: offloaded; the outcome lives in the stored artifact and
    # `result` carries the bounded preview + skim guidance.
    if is_artifact:
        metadata.update({"artifact": True, "tool_call_id": tool_call_id})
        if preview is not None:
            metadata["preview"] = preview
        else:
            metadata["summarized"] = True  # legacy path: no preview supplied
        if isinstance(artifact, dict):
            return _structured_envelope(tool, artifact, result, metadata)
        return build_tool_result(tool, True, result, None, metadata)

    # error-string convention used by the file/web/artifact tools.
    if isinstance(result, str) and _ERROR_PREFIX_RE.match(result):
        return build_tool_result(tool, False, "", result, metadata)

    # default: plain successful output.
    return build_tool_result(tool, True, result, None, metadata)

"""Canonical tool-result envelope builders.

Turns a tool's raw return into the agent-facing envelope whose shape is defined
in :mod:`schema` — including the auto-triggered paths (artifact previews, skips,
parse/arg errors, raised exceptions) — so success/failure is uniform and
machine-checkable, modeled on bash's ``ok``/``exit_code``. ``output`` carries the
raw payload (stdout / file content / report / summary); tool-specific extras go
in the flexible ``metadata`` bag.

This module is pure data logic: no LLM client, no execution state, no I/O. The
``ToolExecutor`` composes it, injecting the stored artifact into
:func:`result_envelope`.
"""

import json
from typing import Any

from .schema import (
    _ERROR_PREFIX_RE,
    _TOOL_RESULT_KEYS,
    DUPLICATE_OF_KEYS,
    TOOL_SCHEMA_VERSION,
)

# Tools whose raw output is the bash-style structured contract
# (`ok`/`stdout`/`stderr`/`exit_code`). Only these are eligible for the
# `_maybe_json` sniff in `result_envelope` — sniffing by content shape alone
# would let an arbitrary JSON payload (e.g. a fetched body containing an
# `"ok"` key) be misinterpreted as this contract.
_STRUCTURED_RESULT_TOOLS = frozenset({"bash"})


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
    _validate_duplicate_of(result["metadata"].get("duplicate_of"))
    return result


def _validate_duplicate_of(duplicate_of: Any) -> None:
    """Enforce the ``metadata.duplicate_of`` shape when present (absent is fine)."""
    if duplicate_of is None:
        return
    if not isinstance(duplicate_of, dict):
        raise ValueError("tool result 'metadata.duplicate_of' must be a dict")
    for stream, descriptor in duplicate_of.items():
        if stream not in ("stdout", "stderr"):
            raise ValueError(
                f"tool result 'metadata.duplicate_of' has invalid stream key: "
                f"{stream!r}"
            )
        missing_keys = [k for k in DUPLICATE_OF_KEYS if k not in descriptor]
        if missing_keys:
            raise ValueError(
                f"tool result 'metadata.duplicate_of[{stream!r}]' missing "
                f"keys: {missing_keys}"
            )


def _maybe_json(text: Any) -> Any:
    """Parse ``text`` as JSON, returning None when it is not JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _structured_envelope(
    tool: str, data: dict[str, Any], output: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build an envelope from a bash-style ``{ok, exit_code, stderr}`` result.

    One field per datum, nothing duplicated: ``output`` is stdout, ``error`` is
    stderr whenever non-empty (regardless of ``ok`` — so a masked pipeline
    failure surfaces as ``ok:true`` with a non-null ``error``, and stderr is
    never dropped), ``exit_code`` lives only in ``metadata``, and ``ok`` is the
    sole success verdict. stderr is never copied into ``metadata``.
    """
    ok = bool(data.get("ok", True))
    if "exit_code" in data:
        metadata["exit_code"] = data["exit_code"]
    error = data.get("stderr") or None
    return build_tool_result(tool, ok, output, error, metadata)


def result_envelope(
    tool: str,
    result: str,
    is_artifact: bool,
    is_truncated: bool,
    tool_call_id: str,
    artifact: Any = None,
    preview: dict[str, Any] | None = None,
    error: str | None = None,
    duplicate_of: dict[str, Any] | None = None,
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
        artifact: The stored full artifact dict for this call (injected by the
            executor so this function stays pure). Required when ``is_artifact``.
        preview: Per-stream ``preview`` descriptor for an offloaded artifact
            (``{"stdout": {...}, "stderr": {...}}``, a key per previewed stream).
            When non-empty it is attached to ``metadata.preview``.
        error: The composed ``error`` (stderr) for an offloaded artifact — a bounded
            preview when the stderr was large, the inline stderr when small, or
            ``None`` when empty. Ignored on non-artifact paths.
        duplicate_of: Per-stream ``{"stdout": {...}, "stderr": {...}}`` descriptor
            (a key per stream whose offload-bound output deduplicated against an
            already-stored artifact). When non-empty it is attached to
            ``metadata.duplicate_of``.
    """
    metadata: dict[str, Any] = {}
    if is_truncated:
        metadata["truncated"] = True

    # bash-family: structured JSON carrying its own ok/exit_code. Gated by tool
    # name, not content shape, so an arbitrary fetched JSON body containing an
    # "ok" key is never reinterpreted as this contract.
    if tool in _STRUCTURED_RESULT_TOOLS:
        parsed = _maybe_json(result)
        if isinstance(parsed, dict) and "ok" in parsed:
            output = parsed.get("stdout", "")
            return _structured_envelope(tool, parsed, output, metadata)

    # large output offloaded per stream — `result` is the composed stdout
    # (preview or inline) and `error` is the composed stderr; each large stream's
    # full content lives in its own on-disk file described by `preview`. Any
    # extra `metadata` bag on the structured-return dict rides along untouched.
    if is_artifact:
        metadata.update(artifact.get("metadata") or {})
        if preview:
            metadata["artifact"] = True
            metadata["tool_call_id"] = tool_call_id
            metadata["preview"] = preview
        if duplicate_of:
            metadata["artifact"] = True
            metadata["tool_call_id"] = tool_call_id
            metadata["duplicate_of"] = duplicate_of
        ok = bool(artifact.get("ok", True))
        if "exit_code" in artifact:
            metadata["exit_code"] = artifact["exit_code"]
        return build_tool_result(tool, ok, result, error, metadata)

    # error-string convention used by the file/web/artifact tools.
    if isinstance(result, str) and _ERROR_PREFIX_RE.match(result):
        return build_tool_result(tool, False, "", result, metadata)

    # default: plain successful output.
    return build_tool_result(tool, True, result, None, metadata)

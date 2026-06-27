"""Tool-call argument preparation: parse, alias-remap, strip unknown kwargs.

Pure helpers the executor composes before dispatch — no client and no execution
state. Each takes the executor's logger so log output stays attributed to it.
"""

import inspect
import json

from ..observability import get_logger
from ..tool_registry import ToolRegistry
from ..utils.parsing import parse_tool_args

logger = get_logger(__name__)

# Known parameter aliases: maps wrong arg name → correct arg name per tool.
# Handles recurring model hallucinations
# (e.g. bash(path=) instead of bash(command=)).
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "bash": {
        "path": "command",
        "cmd": "command",
        "script": "command",
        "shell": "command",
    },
    "read_file": {
        "path": "file_path",
        "filename": "file_path",
        "filepath": "file_path",
    },
    "write_file": {
        "path": "file_path",
        "filename": "file_path",
        "filepath": "file_path",
    },
}


def parse_tool_call(
    tool_call: dict,
) -> tuple[str, str | None, dict | None, str | None]:
    """Parse and validate a raw tool_call dict from the LLM response.

    Returns (tool_call_id, func_name, args, error). error is None on success;
    func_name is set on JSON-parse failure for record creation.
    """
    tool_call_id = tool_call.get("id", "unknown_id")

    tool_type = tool_call.get("type")
    if tool_type is None:
        logger.warning(
            "skip %s — malformed tool call: missing 'type' field", tool_call_id
        )
        return (
            tool_call_id,
            None,
            None,
            "Error: malformed tool call — missing 'type' field",
        )
    if tool_type != "function":
        logger.warning("skip %s — type '%s' not supported", tool_call_id, tool_type)
        return (
            tool_call_id,
            None,
            None,
            f"Error: tool type '{tool_type}' is not supported",
        )

    func_block = tool_call.get("function")
    func_name = func_block.get("name") if func_block else None
    if not func_name:
        logger.warning(
            "skip %s — malformed tool call: missing 'function.name'", tool_call_id
        )
        return (
            tool_call_id,
            None,
            None,
            "Error: malformed tool call — missing 'function.name'",
        )

    # func_name is truthy here, which is only possible when func_block is truthy.
    assert func_block is not None
    try:
        args = parse_tool_args(func_block.get("arguments", {}))
    except json.JSONDecodeError as exc:
        logger.error("malformed args %s → %s: %s", tool_call_id, func_name, exc)
        return (
            tool_call_id,
            func_name,
            None,
            f"Error: could not parse tool arguments as JSON: {exc}",
        )

    return tool_call_id, func_name, args, None


def apply_arg_aliases(func_name: str, args: dict) -> dict:
    """Remap known wrong parameter names to their correct names for func_name."""
    for wrong, correct in _ARG_ALIASES.get(func_name, {}).items():
        if wrong in args and correct not in args:
            logger.warning(
                "arg alias: %s(%s=) → %s(%s=)", func_name, wrong, func_name, correct
            )
            args[correct] = args.pop(wrong)
    return args


def strip_unknown_args(func_name: str, args: dict) -> dict:
    """Drop kwargs not in the tool's signature, logging each dropped arg.

    This prevents TypeError from hallucinated parameters (e.g. file_path on bash)
    from ever reaching the LLM correction loop, which is unreliable on local models.
    """
    func = getattr(ToolRegistry, func_name, None)
    if func is None:
        return args
    valid = set(inspect.signature(func).parameters)
    dropped = {k: v for k, v in args.items() if k not in valid}
    if dropped:
        for k in dropped:
            logger.warning(
                "stripped unknown arg: %s(%s=) — not in tool signature",
                func_name,
                k,
            )
        args = {k: v for k, v in args.items() if k in valid}
    return args

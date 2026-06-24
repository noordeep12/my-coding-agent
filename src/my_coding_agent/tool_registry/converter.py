"""Convert Python functions into OpenAI-compatible tool definitions.

Parses Google-style docstrings (Args:, Tags: sections) and reflects on the
function signature to build the JSON schema expected by the LLM API.
"""

import inspect
import re
from typing import Any, Callable

from ..utils.exceptions import ToolDefinitionError


def _parse_tags_section(docstring: str) -> list[str]:
    """Extract tags from a Google-style Tags: section (comma-separated on one line)."""
    if not docstring:
        return []
    m = re.search(r"\bTags:\s*\n\s*(.+)", docstring)
    if not m:
        return []
    return [t.strip().lower() for t in m.group(1).split(",") if t.strip()]


def _strip_tags_section(docstring: str) -> str:
    """Return the docstring with the Tags: section removed."""
    return re.sub(r"\s*\bTags:\s*\n\s*.+", "", docstring, flags=re.DOTALL).strip()


def _parse_args_section(docstring: str) -> dict[str, str]:  # noqa: C901
    """Extract {param: description} from a Google-style Args: section."""
    if not docstring:
        return {}
    m = re.search(r"\bArgs:\s*\n(.*?)(?:\n\s*\n\S|\Z)", docstring, re.DOTALL)
    if not m:
        return {}
    block = m.group(1)
    first_param = re.search(r"^(\s+)\w+:", block, re.MULTILINE)
    if not first_param:
        return {}
    param_indent = len(first_param.group(1))
    continuation_indent = param_indent + 1
    result: dict[str, str] = {}
    current_param: str | None = None
    current_lines: list[str] = []
    for line in block.splitlines():
        if not line.strip():
            continue
        stripped = line.lstrip()
        line_indent = len(line) - len(stripped)
        if line_indent == param_indent:
            param_match = re.match(r"(\w+):\s*(.*)", stripped)
            if param_match:
                if current_param:
                    result[current_param] = " ".join(current_lines).strip()
                current_param = param_match.group(1)
                current_lines = [param_match.group(2)]
                continue
        if current_param and line_indent >= continuation_indent:
            current_lines.append(stripped)
    if current_param:
        result[current_param] = " ".join(current_lines).strip()
    return result


def _strip_args_section(docstring: str) -> str:
    """Return the docstring with the Args: and Tags: sections removed."""
    cleaned = re.sub(r"\s*\bArgs:\s*\n.*", "", docstring, flags=re.DOTALL)
    cleaned = re.sub(r"\s*\bTags:\s*\n\s*.+", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def function_to_json(func: Callable[..., Any]) -> dict:
    """Convert a Python function into an OpenAI-compatible tool definition dict."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ToolDefinitionError(
            f"Failed to get signature for function {func.__name__}: {e}",
            hint="Tool functions must have an inspectable signature.",
        ) from e

    docstring = inspect.cleandoc(func.__doc__ or "")
    param_descriptions = _parse_args_section(docstring)
    tags = _parse_tags_section(docstring)
    top_description = _strip_args_section(docstring)

    parameters = {}
    required = []
    for param in signature.parameters.values():
        if param.name in ("self", "cls"):
            continue
        param_type = type_map.get(param.annotation, "string")
        entry: dict = {"type": param_type}
        if param.name in param_descriptions:
            entry["description"] = param_descriptions[param.name]
        parameters[param.name] = entry
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": top_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
        "tags": tags,
    }


def tool(func: Callable[..., Any]) -> dict:
    """Decorator/converter: turn a Python function into an LLM tool definition."""
    return function_to_json(func)

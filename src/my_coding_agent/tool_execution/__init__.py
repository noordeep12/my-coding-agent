"""Tool-call dispatch for one assistant message.

Defines ``ToolExecutor``, constructed per message: it parses and validates each
raw tool call, applies argument aliases and strips unknown kwargs, dispatches
through the ``ToolRegistry``, and separates oversized outputs into artifacts —
described deterministically. It makes no LLM calls itself; the LLM client is held
only for the session log path and the observability recorder.
"""

import inspect
import json
import subprocess
from typing import TYPE_CHECKING, Any

from ..observability import get_logger
from ..tool_registry import ToolRegistry
from . import args as arg_prep
from .output import (
    MAX_TOOL_OUTPUT_CHARS,
    _extract_summary,
    describe_artifact,
    validate_tool_output,
)
from .records import call_record, error_record
from .result_schema import (
    TOOL_SCHEMA_VERSION,
    build_tool_result,
    result_envelope,
    validate_tool_result,
)

if TYPE_CHECKING:
    from ..llm import LLM

__all__ = [
    "ToolExecutor",
    "ToolRegistry",
    "MAX_TOOL_OUTPUT_CHARS",
    "TOOL_SCHEMA_VERSION",
    "build_tool_result",
    "validate_tool_result",
    "_extract_summary",
]

# Exceptions a tool may raise that are surfaced as an ``ok:false`` result rather
# than re-raised. Anything not in this tuple hard-stops the agent loop.
_RECOVERABLE_EXCEPTIONS = (
    TypeError,  # wrong arg names / types
    ValueError,  # bad arg values
    FileNotFoundError,  # wrong path
    json.JSONDecodeError,  # malformed tool arguments
    subprocess.TimeoutExpired,  # belt-and-suspenders (bash catches this itself)
)

# Data contract, output post-processing, and argument prep live in the sibling
# modules result_schema / output / args; the executor below composes them. The
# canonical schema (build/validate/envelope), the truncation limit
# (MAX_TOOL_OUTPUT_CHARS) and _extract_summary are imported above.


class ToolExecutor:
    """Dispatch the tool calls in one assistant message.

    Constructed per message: holds that message's ``tool_calls`` plus the running
    ``tool_messages`` / ``tool_records`` it fills and the ``tool_artifacts`` it
    offloads. It owns no LLM calls — the LLM client is kept only for the session
    log path and the observability recorder (``llm._recorder``).
    """

    def __init__(self, message: dict[str, Any], llm: "LLM") -> None:
        self.tool_calls = message.get("tool_calls", []) or []
        self.tool_messages: list[dict[str, Any]] = []
        self.tool_records: list[dict[str, Any]] = []
        self.tool_artifacts: dict = {}
        self.llm = llm
        self.logger = get_logger(self.__class__.__name__)
        self.registry = ToolRegistry(artifacts=self.tool_artifacts)

    def run(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Dispatch every tool call, filling ``tool_messages`` / ``tool_records``.

        Each call runs the three phases — before → call → after — except parse
        failures, which short-circuit to an error result. Returns the two lists
        for convenience; they are also available as attributes.
        """
        self.logger.tool("dispatch: %d tool call(s)", len(self.tool_calls))
        for tool_call in self.tool_calls:
            tool_call_id, func_name, args, error = arg_prep.parse_tool_call(tool_call)
            if error:
                name = func_name or "<unknown>"
                env = build_tool_result(
                    name, False, "", error, {"reason": "parse_error"}
                )
                self.tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(validate_tool_result(env), default=str),
                        "status": "error",
                    }
                )
                self.tool_records.append(error_record(name, {}, tool_call_id, error))
                continue

            # parse_tool_call guarantees func_name/args are set when error is None.
            assert func_name is not None and args is not None

            args = self.before_tool_call(func_name, args)
            self.logger.tool("%s → %s(%s)", tool_call_id, func_name, args)
            raw, failure = self.invoke_tool(tool_call_id, func_name, args)
            content, status, record = self.after_tool_call(
                tool_call_id, func_name, args, raw, failure
            )
            self.tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                    "status": status,
                }
            )
            self.tool_records.append(record)

        return self.tool_messages, self.tool_records

    def before_tool_call(self, func_name: str, args: dict) -> dict:
        """Before the call: alias-remap args, strip unknown kwargs, stamp recorder.

        Returns the prepared args. The recorder (if any) stamps the call's start
        time for latency accounting.
        """
        args = arg_prep.apply_arg_aliases(func_name, args)
        args = arg_prep.strip_unknown_args(func_name, args)
        self.logger.tool("before %s(%s) [after alias remapping]", func_name, args)
        if self.llm._recorder is not None:
            self.llm._recorder.before_tool(func_name, args)
        return args

    def invoke_tool(
        self, tool_call_id: str, func_name: str, args: dict
    ) -> tuple[Any, dict | None]:
        """The call step only: invoke ``func_name(**args)`` against the registry.

        Returns ``(raw_result, failure)`` — the tool's raw return value (str or
        artifact tuple) and ``None`` on success, or ``None`` and a
        ``{"reason", "error"}`` descriptor on a handled failure. No retries and no
        LLM: a wrong-argument call fails directly. Non-recoverable exceptions
        re-raise. Turning the raw result into the envelope is
        :meth:`after_tool_call`'s job.
        """
        if not hasattr(self.registry, func_name):
            self.logger.error("not found: '%s' is not registered", func_name)
            valid = [n for n in dir(ToolRegistry) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            return None, {"reason": "not_found", "error": err}

        try:
            return getattr(self.registry, func_name)(**args), None
        except TypeError as exc:  # wrong arguments — surfaced as a failure, no retry
            sig = inspect.signature(getattr(ToolRegistry, func_name))
            self.logger.error("wrong args %s → %s: %s", tool_call_id, func_name, exc)
            err = (
                f"Error: wrong arguments for '{func_name}': {exc}. "
                f"Expected: {func_name}{sig}"
            )
            return None, {"reason": "wrong_args", "error": err}
        except Exception as exc:
            if not isinstance(exc, _RECOVERABLE_EXCEPTIONS):
                self.logger.error(
                    "non-recoverable error %s → %s: %s", tool_call_id, func_name, exc
                )
                raise
            self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
            err = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
            return None, {"reason": "raised", "error": err}

    def after_tool_call(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        raw_result: Any,
        failure: dict | None,
    ) -> tuple[str, str, dict]:
        """Turn the tool's raw return (or failure) into (content, status, record).

        On failure, builds the error envelope from the ``{reason, error}``
        descriptor. On success, offloads artifact tuples (described
        deterministically — no LLM), coerces to str, truncates, and normalizes
        into the canonical envelope. Serializes, then lets the recorder capture
        the final agent-facing content.
        """

        def capture(content: str) -> str:
            """Let the observability recorder (if any) emit the tool event."""
            if self.llm._recorder is not None:
                self.llm._recorder.after_tool(func_name, args, content)
            return content

        if failure is not None:
            env = build_tool_result(
                func_name, False, "", failure["error"], {"reason": failure["reason"]}
            )
            status = "error"
            record = error_record(func_name, args, tool_call_id, failure["error"])
        else:
            is_artifact = isinstance(raw_result, tuple) and len(raw_result) == 2
            if is_artifact:
                _, artifact = raw_result
                self.tool_artifacts[tool_call_id] = artifact
                result = describe_artifact(artifact, tool_call_id)
            else:
                result = raw_result
            if not isinstance(result, str):
                result = str(result)
            pre_len = len(result)
            result = validate_tool_output(
                result, func_name, self.llm._session_log_path, is_summary=is_artifact
            )
            is_truncated = not is_artifact and len(result) < pre_len

            self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
            env = result_envelope(
                func_name,
                result,
                is_artifact,
                is_truncated,
                tool_call_id,
                self.tool_artifacts.get(tool_call_id),
            )
            status, record = call_record(
                func_name, args, tool_call_id, env, is_artifact, is_truncated
            )

        serialized = json.dumps(validate_tool_result(env), default=str)
        return capture(serialized), status, record

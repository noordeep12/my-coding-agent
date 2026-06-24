"""Tool-call dispatch, argument correction, and artifact separation.

Defines ``ToolExecutor``, which holds the LLM client and the ``tool_artifacts``
store and runs every tool call requested in an LLM response: it parses and
validates each raw call, applies argument aliases and strips unknown kwargs,
runs the before/after hooks, dispatches through the ``ToolsRegistry`` with an
LLM-driven argument-correction retry loop, and separates oversized outputs into
artifacts with an LLM-generated summary returned to the model.
"""

import inspect
import json
import subprocess
from typing import TYPE_CHECKING, Any

from ..logger import get_logger
from ..observability.records import call_record, error_record
from ..tools import ToolsRegistry
from . import args as arg_prep
from .output import (
    MAX_TOOL_OUTPUT_CHARS,
    _extract_summary,
    summarize_artifact,
    validate_tool_output,
)
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
    "ToolsRegistry",
    "MAX_TOOL_OUTPUT_CHARS",
    "TOOL_SCHEMA_VERSION",
    "build_tool_result",
    "validate_tool_result",
    "_extract_summary",
]

# Exceptions the LLM can recover from — returned as error content, not re-raised.
# Anything not in this tuple hard-stops the agent loop via re-raise.
_RECOVERABLE_EXCEPTIONS = (
    TypeError,  # wrong arg names / types — LLM can fix
    ValueError,  # bad arg values — LLM can fix
    FileNotFoundError,  # wrong path — LLM can fix
    json.JSONDecodeError,  # malformed tool arguments — LLM can fix
    subprocess.TimeoutExpired,  # belt-and-suspenders (bash catches this itself)
)

# Max retries for the inner arg-correction loop before falling back to error result.
_MAX_ARG_RETRIES: int = 3

# Data contract, output post-processing, and argument prep live in the sibling
# modules result_schema / output / args; the executor below composes them. The
# canonical schema (build/validate/envelope), the truncation limit
# (MAX_TOOL_OUTPUT_CHARS) and _extract_summary are imported above.


class ToolExecutor:
    """Dispatch tool calls for the agent loop, holding the client and artifacts.

    Own the ``tool_artifacts`` store (execution state) and the LLM client used
    for argument-correction retries and artifact summarization. Parse each raw
    tool call, apply argument aliases, strip unknown kwargs, run the before/after
    hooks, dispatch through ``ToolsRegistry``, and separate oversized outputs into
    artifacts summarized for the model.
    """

    def __init__(self, client: "LLM") -> None:
        """Hold the LLM client and initialize the artifact store.

        Args:
            client: The LLM client used for the argument-correction call and the
                artifact-summarization call, and whose ``_session_log_path`` and
                hooks the executor reads when dispatching tools.
        """
        self.client = client
        self.tool_artifacts: dict = {}
        self.logger = get_logger(self.__class__.__name__)

    def before_tool_call(
        self, tool_call_id: str, func_name: str, args: dict
    ) -> dict | None:
        """Runs before every tool dispatch: alias-remap args, then apply the user hook.

        Returns the (possibly modified) args to proceed, or None to skip the call.
        """
        args = arg_prep.apply_arg_aliases(func_name, args)
        args = arg_prep.strip_unknown_args(func_name, args)
        self.logger.tool(
            "%s → before_hook %s(%s) [after alias remapping]",
            tool_call_id,
            func_name,
            args,
        )
        result = self.client._before_hook(func_name, args)
        if result is None:
            self.logger.tool("%s → before_hook skipped %s", tool_call_id, func_name)
        elif result != args:
            self.logger.tool(
                "%s → before_hook modified %s args: %s", tool_call_id, func_name, result
            )
        return result

    def invoke_tool(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        registry: ToolsRegistry,
        tool_call: dict,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[Any, dict | None]:
        """Call ``func_name(**args)`` with LLM arg-correction retries.

        This is the call step only: it returns ``(raw_result, failure)`` — on
        success the tool's raw return value (str or artifact tuple) and ``None``;
        on a handled failure ``None`` and a ``{"reason", "error"}`` descriptor.
        Turning the raw result into the canonical envelope is
        :meth:`after_tool_call`'s job. Non-recoverable exceptions re-raise.
        """
        if not hasattr(registry, func_name):
            self.logger.error("not found: '%s' is not registered", func_name)
            valid = [n for n in dir(ToolsRegistry) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            return None, {"reason": "not_found", "error": err}

        sig = inspect.signature(getattr(ToolsRegistry, func_name))

        for attempt in range(_MAX_ARG_RETRIES + 1):
            try:
                return getattr(registry, func_name)(**args), None

            except TypeError as wrong_args_exc:  # wrong arguments — ask the LLM to fix
                self.logger.error(
                    "wrong args %s → %s (attempt %s/%s): %s",
                    tool_call_id,
                    func_name,
                    attempt + 1,
                    _MAX_ARG_RETRIES,
                    wrong_args_exc,
                )
                retries_exhausted = attempt == _MAX_ARG_RETRIES
                corrected_args = (
                    None
                    if retries_exhausted
                    else arg_prep.correct_args(
                        self.client,
                        func_name,
                        args,
                        wrong_args_exc,
                        sig,
                        tool_call,
                        tool_call_id,
                        attempt,
                        conversation,
                        tools,
                    )
                )
                if corrected_args is None:
                    err = (
                        f"Error: wrong arguments for '{func_name}' after "
                        f"{attempt + 1} attempt(s): {wrong_args_exc}. "
                        f"Expected: {func_name}{sig}"
                    )
                    return None, {"reason": "wrong_args", "error": err}
                args = corrected_args

            # other errors — return as a failure (don't re-raise, so the agent
            # can keep going); non-recoverable ones re-raise.
            except Exception as exc:
                if not isinstance(exc, _RECOVERABLE_EXCEPTIONS):
                    self.logger.error(
                        "non-recoverable error %s → %s: %s",
                        tool_call_id,
                        func_name,
                        exc,
                    )
                    raise
                self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
                err = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                return None, {"reason": "raised", "error": err}

        # Unreachable: the final iteration always returns (success, or a failure
        # once retries are exhausted). Present so mypy can prove the function returns.
        raise AssertionError("invoke_tool retry loop exited without returning")

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
        descriptor. On success, offloads+summarizes artifact tuples, coerces to
        str, truncates, and normalizes into the canonical envelope. Then
        serializes and applies the user after-hook **last**, so the recorder
        captures the final agent-facing content.
        """
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
                result = summarize_artifact(
                    self.client, artifact, func_name, tool_call_id
                )
            else:
                result = raw_result
            if not isinstance(result, str):
                result = str(result)
            pre_len = len(result)
            result = validate_tool_output(
                result, func_name, self.client._session_log_path, is_summary=is_artifact
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
        content = self._apply_after_hook(tool_call_id, func_name, args, serialized)
        return content, status, record

    def _apply_after_hook(
        self, tool_call_id: str, func_name: str, args: dict, result: str
    ) -> str:
        """Apply the user after-hook to the serialized result (recorder capture)."""
        self.logger.tool(
            "%s → after_hook %s(%s) → %s", tool_call_id, func_name, args, result
        )
        try:
            modified = self.client._after_hook(func_name, args, result)
        except Exception as exc:
            self.logger.error(
                "%s → after_hook raised %s for %s: %s",
                tool_call_id,
                type(exc).__name__,
                func_name,
                exc,
            )
            return result
        if modified != result:
            self.logger.tool(
                "%s → after_hook modified result for %s", tool_call_id, func_name
            )
        return modified

    @staticmethod
    def _emit(
        messages: list[dict[str, Any]],
        records: list[dict[str, Any]],
        tool_call_id: str,
        content: str,
        status: str,
        record: dict[str, Any],
    ) -> None:
        """Append one tool-result message and its matching call-record."""
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "status": status,
            }
        )
        records.append(record)

    def execute_tool_calls(
        self,
        message: dict[str, Any],
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Dispatch all tool calls in message, returning (tool_messages, call_records).

        Args:
            message: The assistant message whose ``tool_calls`` are dispatched.
            conversation: The conversation so far, threaded into the
                argument-correction call (replaces the former ``self.messages``
                reach-back).
            tools: Tool schemas to expose during argument correction (replaces the
                former ``self.tools`` reach-back).

        Success record: {"name": str, "args": dict, "ok": True}
        Failure record: {"name": str, "args": dict, "ok": False, "error": str}
        """
        tool_calls = message.get("tool_calls", []) or []
        messages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        registry = ToolsRegistry(artifacts=self.tool_artifacts, tools=tools or [])
        self.logger.tool("dispatch: %d tool call(s)", len(tool_calls))

        for tool_call in tool_calls:
            # Parse and validate the raw tool call first, to catch issues
            # before invoking any tools.
            tool_call_id, func_name, args, error = arg_prep.parse_tool_call(tool_call)
            if error:
                name = func_name or "<unknown>"
                env = build_tool_result(
                    name, False, "", error, {"reason": "parse_error"}
                )
                record = error_record(name, {}, tool_call_id, error)
                content = json.dumps(validate_tool_result(env), default=str)
                self._emit(messages, records, tool_call_id, content, "error", record)
                continue

            # error is None here, so parse_tool_call gave a valid func_name and args.
            assert func_name is not None and args is not None

            # Run the before_tool_call hook, which can modify args or return
            # None to skip the call.
            args = self.before_tool_call(tool_call_id, func_name, args)
            if args is None:
                env = build_tool_result(
                    func_name,
                    False,
                    "(tool call skipped)",
                    "skipped",
                    {"reason": "skipped"},
                )
                record = error_record(
                    func_name, {}, tool_call_id, "skipped", status="skipped"
                )
                content = json.dumps(validate_tool_result(env), default=str)
                self._emit(messages, records, tool_call_id, content, "skipped", record)
                continue

            # Call the tool (with arg-correction retries), then post-process its
            # raw result (or failure) into the final message + record.
            self.logger.tool("%s → %s(%s)", tool_call_id, func_name, args)
            raw, failure = self.invoke_tool(
                tool_call_id,
                func_name,
                args,
                registry,
                tool_call,
                conversation,
                tools,
            )
            content, status, record = self.after_tool_call(
                tool_call_id, func_name, args, raw, failure
            )
            self._emit(messages, records, tool_call_id, content, status, record)

        return messages, records

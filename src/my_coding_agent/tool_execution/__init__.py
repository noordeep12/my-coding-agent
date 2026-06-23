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
from ..tools import ToolsRegistry
from ..utils import extract_message, parse_tool_args
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

    def after_tool_call(
        self, tool_call_id: str, func_name: str, args: dict, result: str
    ) -> str:
        """Runs after every tool dispatch: apply the user hook to the result."""
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

    def _dispatch_tool(
        self, registry: ToolsRegistry, func_name: str, args: dict, tool_call_id: str
    ) -> tuple[str, bool, bool]:
        """Call func_name(**args), handle artifact tuples, coerce and validate.

        Returns (result, is_artifact, is_truncated).
        """
        result = getattr(registry, func_name)(**args)
        is_artifact = isinstance(result, tuple) and len(result) == 2
        if is_artifact:
            _, artifact = result
            self.tool_artifacts[tool_call_id] = artifact
            result = summarize_artifact(self.client, artifact, func_name, tool_call_id)
        if not isinstance(result, str):
            result = str(result)
        pre_len = len(result)
        result = validate_tool_output(
            result, func_name, self.client._session_log_path, is_summary=is_artifact
        )
        is_truncated = not is_artifact and len(result) < pre_len
        return result, is_artifact, is_truncated

    def _correct_args(
        self,
        func_name: str,
        args: dict,
        exc: Exception,
        sig: inspect.Signature,
        tool_call: dict,
        tool_call_id: str,
        attempt: int,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict | None:
        """Ask the LLM to fix wrong args after a TypeError.

        Returns corrected args, or None on failure.
        """
        correction_messages = list(conversation) + [
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call_id, "content": f"Error: {exc}"},
            {
                "role": "user",
                "content": (
                    f"Tool '{func_name}' was called with wrong arguments: {exc}. "
                    f"Expected signature: {func_name}{sig}. "
                    f"Please call '{func_name}' again with the correct arguments."
                ),
            },
        ]
        correction_resp = self.client.chat_completion(
            correction_messages,
            tools=tools,
            kind="tool_arg_correction",
        )
        corrected = next(
            (
                c
                for c in (extract_message(correction_resp).get("tool_calls") or [])
                if c.get("function", {}).get("name") == func_name
            ),
            None,
        )
        if not corrected:
            self.logger.warning(
                "correction attempt %s: model did not return a %s call",
                attempt + 1,
                func_name,
            )
            return None
        try:
            args = parse_tool_args(corrected.get("function", {}).get("arguments", {}))
        except json.JSONDecodeError:
            self.logger.warning(
                "correction attempt %s: could not parse corrected args", attempt + 1
            )
            return None
        args = arg_prep.apply_arg_aliases(func_name, args)
        self.logger.tool(
            "corrected args (attempt %s): %s(%s)", attempt + 1, func_name, args
        )
        return args

    def invoke_tool(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        registry: ToolsRegistry,
        tool_call: dict,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict, str, dict]:
        """Dispatch func_name with arg-correction retries.

        Returns (result_envelope, status, record). The envelope is the canonical
        tool-output schema (see :func:`build_tool_result`).
        """
        if not hasattr(registry, func_name):
            self.logger.error("not found: '%s' is not registered", func_name)
            valid = [n for n in dir(ToolsRegistry) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            env = build_tool_result(func_name, False, "", err, {"reason": "not_found"})
            return env, "error", self._error_record(func_name, args, tool_call_id, err)

        sig = inspect.signature(getattr(ToolsRegistry, func_name))

        for attempt in range(_MAX_ARG_RETRIES + 1):
            try:
                result, is_artifact, is_truncated = self._dispatch_tool(
                    registry, func_name, args, tool_call_id
                )
                self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
                env = result_envelope(
                    func_name,
                    result,
                    is_artifact,
                    is_truncated,
                    tool_call_id,
                    self.tool_artifacts.get(tool_call_id),
                )
                status, record = self._call_record(
                    func_name, args, tool_call_id, env, is_artifact, is_truncated
                )
                return env, status, record

            except (
                TypeError
            ) as wrong_args_exc:  # wrong arguments — attempt correction with the LLM
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
                    else self._correct_args(
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
                    env = build_tool_result(
                        func_name, False, "", err, {"reason": "wrong_args"}
                    )
                    return (
                        env,
                        "error",
                        self._error_record(
                            func_name, args, tool_call_id, str(wrong_args_exc)
                        ),
                    )
                args = corrected_args

            # other errors — log and return as an error result (don't re-raise,
            # so the agent can keep going)
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
                env = build_tool_result(func_name, False, "", err, {"reason": "raised"})
                return (
                    env,
                    "error",
                    self._error_record(func_name, args, tool_call_id, str(exc)),
                )

        # Unreachable: the final iteration always returns (success, or an error result
        # once retries are exhausted). Present so mypy can prove the function returns.
        raise AssertionError("invoke_tool retry loop exited without returning")

    @staticmethod
    def _error_record(
        func_name: str,
        args: dict,
        tool_call_id: str,
        error: str,
        status: str = "error",
    ) -> dict[str, Any]:
        """Build a failure call-record (for session_data.json / observability)."""
        return {
            "name": func_name,
            "args": args,
            "ok": False,
            "error": error,
            "tool_call_id": tool_call_id,
            "artifact": False,
            "truncated": False,
            "status": status,
        }

    @staticmethod
    def _call_record(
        func_name: str,
        args: dict,
        tool_call_id: str,
        env: dict[str, Any],
        is_artifact: bool,
        is_truncated: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Build the (status, call-record) for a dispatched tool, from its envelope."""
        status = "success" if env["ok"] else "error"
        record = {
            "name": func_name,
            "args": args,
            "ok": env["ok"],
            "tool_call_id": tool_call_id,
            "artifact": is_artifact,
            "truncated": is_truncated,
            "status": status,
        }
        if not env["ok"]:
            record["error"] = env["error"] or "tool reported failure"
        return status, record

    def _finalize_result(self, env: dict[str, Any]) -> str:
        """Validate the envelope and serialize it to the agent-facing JSON string."""
        validate_tool_result(env)
        return json.dumps(env, default=str)

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
                record = self._error_record(name, {}, tool_call_id, error)
                self._emit(
                    messages,
                    records,
                    tool_call_id,
                    self._finalize_result(env),
                    "error",
                    record,
                )
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
                record = self._error_record(
                    func_name, {}, tool_call_id, "skipped", status="skipped"
                )
                self._emit(
                    messages,
                    records,
                    tool_call_id,
                    self._finalize_result(env),
                    "skipped",
                    record,
                )
                continue

            # Invoke the tool with retries for argument correction, and handle
            # any exceptions.
            self.logger.tool("%s → %s(%s)", tool_call_id, func_name, args)
            env, status, record = self.invoke_tool(
                tool_call_id,
                func_name,
                args,
                registry,
                tool_call,
                conversation,
                tools,
            )

            # Serialize to the canonical schema, then post-process before the LLM.
            content = self.after_tool_call(
                tool_call_id, func_name, args, self._finalize_result(env)
            )
            self._emit(messages, records, tool_call_id, content, status, record)

        return messages, records

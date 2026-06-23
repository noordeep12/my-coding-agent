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
import re
import subprocess
from typing import TYPE_CHECKING, Any

from ..logger import get_logger
from ..tools import ARTIFACT_THRESHOLD, ToolsRegistry
from ..utils import extract_message, parse_tool_args
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
]

# Single source of truth lives in tools.ARTIFACT_THRESHOLD: the artifact-separation
# boundary and this truncation boundary are the same concept (large tool output).
MAX_TOOL_OUTPUT_CHARS = ARTIFACT_THRESHOLD

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

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _extract_summary(content: str) -> str:
    """Return only the summary, dropping any model thinking/preamble.

    Prefers the explicit ``<summary>...</summary>`` block the prompt asks for;
    otherwise strips ``<think>...</think>`` reasoning blocks and returns the rest.
    """
    match = _SUMMARY_RE.search(content)
    if match:
        return match.group(1).strip()
    return _THINK_RE.sub("", content).strip()


# The canonical tool-output schema (build/validate/envelope) lives in
# ``result_schema`` — a pure data-contract module the executor composes.

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

    def _validate_tool_output(
        self, result: str, func_name: str, is_summary: bool = False
    ) -> str:
        if not result.strip():
            return "(tool returned empty output)"
        if not is_summary and len(result) > MAX_TOOL_OUTPUT_CHARS:
            log_hint = (
                f" Use read_file(file_path='{self.client._session_log_path}') "
                "to inspect the full output."
                if self.client._session_log_path
                else ""
            )
            self.logger.warning(
                "tool output truncated: %s returned %d chars (limit %d)",
                func_name,
                len(result),
                MAX_TOOL_OUTPUT_CHARS,
            )
            result = (
                result[:MAX_TOOL_OUTPUT_CHARS]
                + f"\n[output truncated at {MAX_TOOL_OUTPUT_CHARS} chars —"
                + f" full output is in the session log.{log_hint}]"
            )
        if func_name == "bash" and not is_summary:
            try:
                json.loads(result.split("\n[output truncated")[0])
            except json.JSONDecodeError:
                self.logger.warning("bash tool returned non-JSON output")
        return result

    def _summarize_artifact(
        self, artifact: dict, func_name: str, tool_call_id: str
    ) -> str:
        self.logger.tool(
            "%s → %s: artifact %s chars (summarizing for model)",
            tool_call_id,
            func_name,
            len(json.dumps(artifact)),
        )
        prompt = (
            "/no_think\n"
            f"Summarize the following `{func_name}` tool output concisely "
            "for an AI coding agent. "
            "Include: exit status, key findings, any errors, and what the "
            "agent needs to know to continue its task. "
            "Be factual and brief — 3 to 8 sentences max.\n"
            "Output ONLY the summary itself — no reasoning, analysis, planning, "
            "or preamble. Wrap the summary in <summary>...</summary> tags.\n\n"
            f"Output:\n{json.dumps(artifact, indent=2)[:12_000]}"
        )
        try:
            resp = self.client.chat_completion(
                [{"role": "user", "content": prompt}],
                tools=[],
                kind="tool_output_summarizer",
                max_tokens=512,
            )
            summary = _extract_summary(extract_message(resp).get("content") or "")
        except Exception as exc:
            self.logger.warning("artifact summarization failed: %s", exc)
            if "content" in artifact:
                summary = json.dumps(
                    {
                        "file_path": artifact.get("file_path"),
                        "size": artifact.get("size"),
                    }
                )
            else:
                summary = json.dumps(
                    {
                        "exit_code": artifact.get("exit_code"),
                        "ok": artifact.get("ok"),
                        "stdout_chars": len(artifact.get("stdout", "")),
                        "stderr_chars": len(artifact.get("stderr", "")),
                    }
                )
        return summary.strip() + (
            f"\n\n[Full output stored as artifact — use "
            f'read_tool_artifact(tool_call_id="{tool_call_id}") ONLY if the '
            "summary above is insufficient to proceed. "
            "Avoid calling it unless strictly necessary.]"
        )

    def parse_tool_call(
        self, tool_call: dict
    ) -> tuple[str, str | None, dict | None, str | None]:
        """Parse and validate a raw tool_call dict from the LLM response.

        Returns (tool_call_id, func_name, args, error).
        error is None on success; func_name is set on JSON-parse failure
        for record creation.
        """
        tool_call_id = tool_call.get("id", "unknown_id")

        tool_type = tool_call.get("type")
        if tool_type is None:
            self.logger.warning(
                "skip %s — malformed tool call: missing 'type' field", tool_call_id
            )
            return (
                tool_call_id,
                None,
                None,
                "Error: malformed tool call — missing 'type' field",
            )
        if tool_type != "function":
            self.logger.warning(
                "skip %s — type '%s' not supported", tool_call_id, tool_type
            )
            return (
                tool_call_id,
                None,
                None,
                f"Error: tool type '{tool_type}' is not supported",
            )

        func_block = tool_call.get("function")
        func_name = func_block.get("name") if func_block else None
        if not func_name:
            self.logger.warning(
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
            self.logger.error(
                "malformed args %s → %s: %s", tool_call_id, func_name, exc
            )
            return (
                tool_call_id,
                func_name,
                None,
                f"Error: could not parse tool arguments as JSON: {exc}",
            )

        return tool_call_id, func_name, args, None

    def _apply_arg_aliases(self, func_name: str, args: dict) -> dict:
        """Remap known wrong parameter names to their correct names for func_name."""
        for wrong, correct in _ARG_ALIASES.get(func_name, {}).items():
            if wrong in args and correct not in args:
                self.logger.warning(
                    "arg alias: %s(%s=) → %s(%s=)", func_name, wrong, func_name, correct
                )
                args[correct] = args.pop(wrong)
        return args

    def _strip_unknown_args(self, func_name: str, args: dict) -> dict:
        """Drop kwargs not in the tool's signature, logging each dropped arg.

        This prevents TypeError from hallucinated parameters (e.g. file_path on bash)
        from ever reaching the LLM correction loop, which is unreliable on local models.
        """
        func = getattr(ToolsRegistry, func_name, None)
        if func is None:
            return args
        valid = set(inspect.signature(func).parameters)
        dropped = {k: v for k, v in args.items() if k not in valid}
        if dropped:
            for k in dropped:
                self.logger.warning(
                    "stripped unknown arg: %s(%s=) — not in tool signature",
                    func_name,
                    k,
                )
            args = {k: v for k, v in args.items() if k in valid}
        return args

    def before_tool_call(
        self, tool_call_id: str, func_name: str, args: dict
    ) -> dict | None:
        """Runs before every tool dispatch: alias-remap args, then apply the user hook.

        Returns the (possibly modified) args to proceed, or None to skip the call.
        """
        args = self._apply_arg_aliases(func_name, args)
        args = self._strip_unknown_args(func_name, args)
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
            result = self._summarize_artifact(artifact, func_name, tool_call_id)
        if not isinstance(result, str):
            result = str(result)
        pre_len = len(result)
        result = self._validate_tool_output(result, func_name, is_summary=is_artifact)
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
        args = self._apply_arg_aliases(func_name, args)
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
                env = self._result_envelope(
                    func_name, result, is_artifact, is_truncated, tool_call_id
                )
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
        func_name: str, args: dict, tool_call_id: str, error: str
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
            "status": "error",
        }

    def _result_envelope(
        self,
        tool: str,
        result: str,
        is_artifact: bool,
        is_truncated: bool,
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Normalize a tool's raw return into the canonical schema envelope.

        Thin adapter over :func:`result_schema.result_envelope`: it injects this
        call's stored artifact so the schema logic stays pure and stateless.
        """
        return result_envelope(
            tool,
            result,
            is_artifact,
            is_truncated,
            tool_call_id,
            self.tool_artifacts.get(tool_call_id),
        )

    def _finalize_result(self, env: dict[str, Any]) -> str:
        """Validate the envelope and serialize it to the agent-facing JSON string."""
        validate_tool_result(env)
        return json.dumps(env, default=str)

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
        messages, records = [], []
        registry = ToolsRegistry(artifacts=self.tool_artifacts, tools=tools or [])
        self.logger.tool("dispatch: %d tool call(s)", len(tool_calls))

        for tool_call in tool_calls:
            # Parse and validate the raw tool call first, to catch issues
            # before invoking any tools.
            tool_call_id, func_name, args, error = self.parse_tool_call(tool_call)
            if error:
                env = build_tool_result(
                    func_name or "<unknown>",
                    False,
                    "",
                    error,
                    {"reason": "parse_error"},
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": self._finalize_result(env),
                        "status": "error",
                    }
                )
                records.append(
                    {
                        "name": func_name or "<unknown>",
                        "args": {},
                        "ok": False,
                        "error": error,
                        "tool_call_id": tool_call_id,
                        "artifact": False,
                        "truncated": False,
                        "status": "error",
                    }
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": self._finalize_result(env),
                        "status": "skipped",
                    }
                )
                records.append(
                    {
                        "name": func_name,
                        "args": {},
                        "ok": False,
                        "error": "skipped",
                        "tool_call_id": tool_call_id,
                        "artifact": False,
                        "truncated": False,
                        "status": "skipped",
                    }
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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                    "status": status,
                }
            )
            records.append(record)

        return messages, records

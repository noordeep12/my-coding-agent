"""Tool-call dispatch for one assistant message.

Defines ``ToolExecutor``, constructed per message: it parses and validates each
raw tool call, applies argument aliases and strips unknown kwargs, dispatches
through the ``ToolRegistry``, and offloads oversized outputs: each is written to
a per-artifact file on disk and replaced in the result by a bounded preview (an
excerpt plus skim guidance). It makes no LLM calls itself; the LLM client is held
only for the session log path and the observability recorder.
"""

import inspect
import json
import subprocess
from typing import TYPE_CHECKING, Any

from ...observability import current_session_id
from ...utils import get_logger
from . import args as arg_prep
from .envelope import (
    build_tool_result,
    result_envelope,
    validate_tool_result,
)
from .lang import resolve_lang
from .output import (
    MAX_TOOL_OUTPUT_CHARS,
    PREVIEW_MAX_CHARS,
    build_stream_preview,
    validate_tool_output,
)
from .records import call_record, error_record
from .schema import TOOL_SCHEMA_VERSION

if TYPE_CHECKING:
    from ..llm import LLM

__all__ = [
    "ToolExecutor",
    "ToolRegistry",
    "MAX_TOOL_OUTPUT_CHARS",
    "TOOL_SCHEMA_VERSION",
    "build_tool_result",
    "validate_tool_result",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve ``ToolRegistry`` so it stays part of this module's public
    surface (``__all__``) without an eager import — see ``ToolExecutor.__init__``
    for why that import must be deferred (breaks a cycle with tool_registry)."""
    if name == "ToolRegistry":
        from ..tool_registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Exceptions a tool may raise that are surfaced as an ``ok:false`` result rather
# than re-raised. Anything not in this tuple hard-stops the agent loop.
_RECOVERABLE_EXCEPTIONS = (
    TypeError,  # wrong arg names / types
    ValueError,  # bad arg values
    FileNotFoundError,  # wrong path
    json.JSONDecodeError,  # malformed tool arguments
    subprocess.TimeoutExpired,  # belt-and-suspenders (bash catches this itself)
)

# Data contract, envelope builders, output post-processing, and argument prep
# live in the sibling modules schema / envelope / output / args; the executor
# below composes them. The envelope builders (build/validate/normalize) and the
# truncation limit (MAX_TOOL_OUTPUT_CHARS) are imported above.


class ToolExecutor:
    """Dispatch the tool calls in one assistant message.

    Constructed per message: holds that message's ``tool_calls`` plus the running
    ``tool_messages`` / ``tool_records`` it fills and the ``tool_artifacts`` it
    offloads. It owns no LLM calls — the LLM client is kept only for the session
    log path and the observability recorder (``llm._recorder``).

    The agent's available ``tools`` are forwarded to the ``ToolRegistry`` so
    toolset-aware tools (notably ``delegate``, which spawns a subagent with the
    parent toolset minus ``delegate``) can read them. Omitting ``tools`` leaves
    the registry with an empty toolset. The ``llm`` client is forwarded the same
    way so ``read_tool_artifact`` can make its bounded extraction call.
    """

    def __init__(
        self,
        message: dict[str, Any],
        llm: "LLM",
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        # Imported lazily (not at module level) to avoid a circular import:
        # tool_registry reads its size-threshold constants from
        # tool_execution.schema, so tool_execution can't eagerly import
        # tool_registry back at module load time.
        from ..tool_registry import ToolRegistry

        self.tool_calls = message.get("tool_calls", []) or []
        self.tool_messages: list[dict[str, Any]] = []
        self.tool_records: list[dict[str, Any]] = []
        self.tool_artifacts: dict = {}
        self.llm = llm
        self.logger = get_logger(self.__class__.__name__)
        self.registry = ToolRegistry(
            artifacts=self.tool_artifacts, tools=tools or [], llm=llm
        )

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
                env["metadata"]["lang"] = resolve_lang(name, {}, env)
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
            valid = [n for n in dir(type(self.registry)) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            return None, {"reason": "not_found", "error": err}

        try:
            return getattr(self.registry, func_name)(**args), None
        except TypeError as exc:  # wrong arguments — surfaced as a failure, no retry
            sig = inspect.signature(getattr(type(self.registry), func_name))
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
        descriptor. On success, offloads artifact tuples (writing each to a
        per-artifact file and replacing it with a bounded preview — no LLM),
        coerces to str, truncates, and normalizes into the canonical envelope.
        Serializes, then lets the recorder capture the final agent-facing content.
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
            preview: dict[str, Any] | None = None
            error: str | None = None
            if is_artifact:
                _, artifact = raw_result
                self.tool_artifacts[tool_call_id] = artifact
                result, error, preview = self._offload_streams(tool_call_id, artifact)
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
                preview=preview,
                error=error,
            )
            status, record = call_record(
                func_name, args, tool_call_id, env, is_artifact, is_truncated
            )

        env["metadata"]["lang"] = resolve_lang(func_name, args, env)
        serialized = json.dumps(validate_tool_result(env), default=str)
        return capture(serialized), status, record

    def _offload_streams(
        self, tool_call_id: str, artifact: dict[str, Any]
    ) -> tuple[str, str | None, dict[str, Any]]:
        """Bound each output stream of an offloaded command artifact independently.

        Returns ``(output, error, preview)``: ``output`` is the composed stdout
        (bounded preview when large, else inline), ``error`` is the composed stderr
        (preview, inline, or ``None`` when empty), and ``preview`` maps each stream
        that was previewed to its descriptor.
        """
        preview: dict[str, Any] = {}
        output, out_desc = self._offload_stream(
            tool_call_id, "stdout", artifact.get("stdout") or ""
        )
        if out_desc is not None:
            preview["stdout"] = out_desc
        error, err_desc = self._offload_stream(
            tool_call_id, "stderr", artifact.get("stderr") or ""
        )
        if err_desc is not None:
            preview["stderr"] = err_desc
        return output, (error or None), preview

    def _offload_stream(
        self, tool_call_id: str, stream: str, text: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Return ``(field_value, preview_descriptor)`` for one output stream.

        Small streams (within the preview budget) are inlined with no descriptor
        and no file; larger streams are written to a per-stream file and replaced
        with a bounded excerpt + skim guidance.
        """
        if len(text) <= PREVIEW_MAX_CHARS:
            return text, None
        path = self._write_artifact_file(tool_call_id, stream, text)
        return build_stream_preview(text, path)

    def _write_artifact_file(
        self, tool_call_id: str, stream: str, text: str
    ) -> str | None:
        """Write a stream's full content to its per-run file so bash can skim it.

        The file lives at
        ``.my_coding_agent/<session>/artifacts/<tool_call_id>.<stream>.txt`` and
        persists for the run, so a later step can inspect it with bash text tools.
        Returns the path, or ``None`` when the session directory or id is
        unavailable (e.g. unit tests invoking the executor without an agent run),
        or when the write itself fails (full disk / permissions) — a failed write
        is logged and downgraded to "no on-disk copy" so offloading continues
        rather than aborting the run.
        """
        from ..tool_registry import artifact_file_path  # lazy: avoids a cycle

        path = artifact_file_path(current_session_id.get(), tool_call_id, stream)
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError as exc:
            # A full disk or bad permissions must not abort the run: offloading
            # and the preview continue without an on-disk copy (the preview
            # guidance falls back to read_tool_artifact when the path is None).
            self.logger.warning(
                "artifact write failed for %s (%s) at %s: %s",
                tool_call_id,
                stream,
                path,
                exc,
            )
            return None
        return str(path)

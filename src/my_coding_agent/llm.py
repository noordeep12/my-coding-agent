"""LLM HTTP client and tool-execution machinery.

Defines ``LLM``, the base class that owns the ``httpx`` session, calls the
OpenAI-compatible ``/chat/completions`` endpoint with retries, and records token
usage per call. It also handles tool selection (``route_tools``), tool dispatch
with argument-correction retries (``execute_tool_calls``/``invoke_tool``), and
artifact separation for oversized tool outputs.
"""

import inspect
import json
import os
import re
import subprocess
import time
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from httpx import Response

from .exceptions import APIResponseError
from .logger import get_logger
from .tools import ARTIFACT_THRESHOLD, ToolsRegistry
from .utils import extract_message, parse_tool_args

load_dotenv()
OMLX_API_URL = os.environ.get("OMLX_API_URL", "http://127.0.0.1:8321/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "changeme")
OMLX_MODEL = os.environ.get("OMLX_MODEL", "Qwen3.6-35B-A3B-4bit")

# Single source of truth lives in tools.ARTIFACT_THRESHOLD: the artifact-separation
# boundary and this truncation boundary are the same concept (large tool output).
MAX_TOOL_OUTPUT_CHARS = ARTIFACT_THRESHOLD

# Tools always included regardless of routing decision.
_BASELINE_TOOLS = {"bash", "read_file", "read_tool_artifact"}


# Default per-request HTTP timeout (seconds) for the LLM API session.
DEFAULT_HTTP_TIMEOUT = 30.0
# Number of attempts for transient (connection/timeout) failures on external calls.
_HTTP_RETRIES = 3
# Base backoff (seconds) between retries; doubles each attempt.
_HTTP_BACKOFF = 0.5
# Errors worth retrying — transient connectivity/timeout, not protocol errors.
_TRANSIENT_HTTP_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)


def _search_bracketed(text: str) -> str:
    """Return the first ``[...]`` span in text, or raise ValueError if absent.

    Used as a JSON-array extraction strategy; raising on no-match lets the caller's
    try/except fall through to the next strategy.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        raise ValueError("no bracketed array found")
    return match.group()


class LLM:
    """HTTP client for an OpenAI-compatible LLM server with tool execution.

    Own the ``httpx`` session and issue chat-completion requests, recording every
    call's token usage in ``self.llm_calls`` tagged by ``kind``. Provide two-phase
    tool routing, tool-call dispatch with argument-correction retries, and artifact
    separation that stores oversized tool outputs in ``self.tool_artifacts`` and
    returns a summary to the model. Construction performs no network I/O: the
    model's context window is probed lazily on first access to ``context_window``.
    """

    # 128k fallback used when the server is unreachable or omits a context length.
    DEFAULT_CONTEXT_WINDOW = 131_072

    def __init__(
        self,
        api_url: str = OMLX_API_URL,
        api_key: str = OMLX_API_KEY,
        model: str = OMLX_MODEL,
        before_tool_call: Callable[..., Any] | None = None,
        after_tool_call: Callable[..., Any] | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        """Initialize the LLM client without performing any network I/O.

        Build the HTTP session and defer the model's context-window probe: it is
        resolved lazily on first access to ``self.context_window`` (see the
        property), so construction never reaches the server.

        Args:
            api_url: Base URL of the OpenAI-compatible API (e.g. ``.../v1``).
            api_key: Bearer token sent on every request.
            model: Model id whose context window is looked up on first use.
            before_tool_call: Optional ``(name, args) -> args | None`` hook run
                before each tool dispatch; returning None skips the call.
            after_tool_call: Optional ``(name, args, result) -> result`` hook run
                after each tool dispatch to post-process the result.
            timeout: Per-request HTTP timeout in seconds for the session.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.logger = get_logger(self.__class__.__name__)
        self.setup_session()
        # Resolved lazily on first read of the context_window property.
        self._context_window: int | None = None
        self._session_log_path: str | None = (
            None  # set by Agent after session dir is created
        )
        self.tool_artifacts: dict = {}
        self.llm_calls: list[dict] = []  # one entry per chat_completion call, in order
        self._before_hook: Callable[[str, dict], dict | None] = before_tool_call or (
            lambda name, args: args
        )
        self._after_hook: Callable[[str, dict, str], str] = after_tool_call or (
            lambda name, args, result: result
        )

    def setup_session(self) -> None:
        """Create the httpx client and apply auth headers and the timeout.

        Assign a fresh ``httpx.Client`` to ``self.session``, set the JSON
        content-type and bearer-auth headers, and apply ``self.timeout`` as the
        per-request timeout. Called once from ``__init__``; calling it again
        replaces the existing session.
        """
        self.session = httpx.Client()
        self.session.headers.update(
            {
                "content-type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            }
        )
        self.session.timeout = self.timeout

    @property
    def context_window(self) -> int:
        """The model's context window in tokens, resolved lazily and cached.

        On first access, probe the server via ``available_models`` to discover the
        window; if the probe fails (server unreachable), fall back to
        ``DEFAULT_CONTEXT_WINDOW`` and cache that so construction-time failures
        never surface here. Subsequent reads return the cached value.
        """
        if self._context_window is None:
            try:
                self.available_models()
            except httpx.HTTPError as exc:
                self.logger.warning(
                    "context-window probe failed (%s) — falling back to %d tokens",
                    exc,
                    self.DEFAULT_CONTEXT_WINDOW,
                )
                self._context_window = self.DEFAULT_CONTEXT_WINDOW
        # available_models sets _context_window on success; the except sets it too.
        assert self._context_window is not None
        return self._context_window

    @context_window.setter
    def context_window(self, value: int | None) -> None:
        """Set the cached context window directly (used by tests and resets)."""
        self._context_window = value

    def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> Response:
        """Issue an HTTP request, retrying transient failures with backoff.

        Retries connection/timeout failures. Non-transient errors (HTTP protocol
        errors, etc.) are not retried. Raises the last transient error if all
        attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(_HTTP_RETRIES):
            try:
                return self.session.request(method, url, **kwargs)
            except _TRANSIENT_HTTP_ERRORS as exc:
                last_exc = exc
                if attempt == _HTTP_RETRIES - 1:
                    break
                backoff = _HTTP_BACKOFF * (2**attempt)
                self.logger.warning(
                    "transient HTTP error on %s %s (attempt %s/%s): %s "
                    "— retrying in %.1fs",
                    method,
                    url,
                    attempt + 1,
                    _HTTP_RETRIES,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        # The loop runs at least once (_HTTP_RETRIES >= 1), so reaching here means a
        # transient error was caught and last_exc is set. assert proves this to mypy.
        assert last_exc is not None
        raise last_exc

    def available_models(self) -> list[str]:
        """Fetch the server's model list and cache ``self.context_window``.

        Issue a GET to ``{api_url}/models`` and read each entry's id. As a side
        effect, look up ``self.model`` in the response and cache its context
        length on ``self._context_window``, falling back to
        ``DEFAULT_CONTEXT_WINDOW`` when the model is absent or reports no context
        length.

        Returns:
            The list of model ids advertised by the server.

        Raises:
            httpx.HTTPError: If the request cannot reach the server after the
                transient-failure retries are exhausted.
        """
        resp = self._request_with_retry("GET", self.api_url + "/models")
        data = resp.json().get("data", [])
        models = [m["id"] for m in data]
        self.logger.api("Models: %s", models)
        self._context_window = self.DEFAULT_CONTEXT_WINDOW
        for m in data:
            if m["id"] == self.model:
                self._context_window = (
                    m.get("context_length")
                    or m.get("max_context_length")
                    or m.get("context_window")
                    or self.DEFAULT_CONTEXT_WINDOW
                )
                break
        self.logger.api(
            "Context window for %s: %s tokens", self.model, self._context_window
        )
        return models

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        kind: str = "main",
        max_tokens: int | None = None,
    ) -> Response:
        """Send one chat-completion request and record its token usage.

        POST ``messages`` and ``tools`` to ``{api_url}/chat/completions``, append
        the call's prompt/completion/total token usage to ``self.llm_calls``
        (tagged with ``kind``), log any reasoning/content, and return the raw
        httpx response. The caller extracts the message via ``extract_message``.

        Args:
            messages: Conversation so far, as OpenAI-style role/content dicts.
            tools: Tool schemas to expose this turn; None sends an empty list.
            kind: Usage tag for accounting (``main``, ``handoff``,
                ``tool_router``, ``tool_output_summarizer``,
                ``tool_arg_correction``).
            max_tokens: Optional cap on generated tokens; omitted when None.

        Returns:
            The raw ``httpx.Response`` from the completions endpoint.

        Raises:
            APIResponseError: If the server returns a non-JSON body.
            httpx.HTTPError: If the request cannot reach the server after the
                transient-failure retries are exhausted.
        """
        call_num = len(self.llm_calls) + 1
        self.logger.api(
            "→ POST %s/chat/completions  [call #%d, kind=%s]",
            self.api_url,
            call_num,
            kind,
        )
        self.logger.debug(
            "Request body: %s",
            json.dumps(
                {"model": self.model, "messages": messages, "tools": tools or []},
                indent=4,
            ),
        )

        body: dict = {"model": self.model, "messages": messages, "tools": tools or []}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        resp = self._request_with_retry(
            "POST",
            self.api_url + "/chat/completions",
            json=body,
        )
        self.logger.api(
            "← %d (%d bytes)  [call #%d, kind=%s]",
            resp.status_code,
            len(resp.content),
            call_num,
            kind,
        )
        try:
            data = resp.json()
        except Exception as exc:
            raise APIResponseError(
                f"API returned non-JSON response (HTTP {resp.status_code}): {exc}. "
                f"Body prefix: {resp.text[:200]!r}",
                hint="Check that api_url points at an OpenAI-compatible endpoint "
                "and the server is healthy.",
            ) from exc
        self.logger.debug("Response body: %s", json.dumps(data, indent=4))

        usage = data.get("usage", {})
        self.llm_calls.append(
            {
                "call": call_num,
                "kind": kind,
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        )
        self.logger.api(
            "call #%d [%s] usage — prompt: %s, completion: %s, total: %s",
            call_num,
            kind,
            f"{usage.get('prompt_tokens', 0):,}",
            f"{usage.get('completion_tokens', 0):,}",
            f"{usage.get('total_tokens', 0):,}",
        )

        try:
            choices = data.get("choices", [])
        except Exception as exc:
            self.logger.error("Failed to parse choices: %s", exc)
            choices = []

        for choice in choices:
            message = choice.get("message", {})
            reasoning = message.get("reasoning_content") or ""
            content = message.get("content") or ""
            if reasoning:
                self.logger.llm("Reasoning: \n\n%s\n", reasoning)
            if content:
                self.logger.llm("Content: \n\n%s\n", content)
        return resp

    def route_tools(
        self, message: str, all_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return the subset of all_tools relevant to message.

        Phase 1 — keyword match against each tool's tags (zero cost).
        Phase 2 — LLM fallback only when phase 1 finds zero tag matches
        across ALL tools.
        Baseline tools (bash, read_file, read_tool_artifact) are always included.
        """
        if not all_tools:
            return all_tools

        text = message.lower()
        baseline = [t for t in all_tools if t["function"]["name"] in _BASELINE_TOOLS]
        non_baseline = [
            t for t in all_tools if t["function"]["name"] not in _BASELINE_TOOLS
        ]

        # Skip routing entirely when there are no non-baseline tools to choose from.
        if not non_baseline:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (no non-baseline tools, skipped)", names
            )
            return all_tools

        # Phase 1: keyword match on tags — check non-baseline tools first.
        keyword_matched = [
            t for t in non_baseline if any(tag in text for tag in t.get("tags", []))
        ]

        if keyword_matched:
            selected = baseline + keyword_matched
            names = [t["function"]["name"] for t in selected]
            self.logger.tool("router phase-1 → %s", names)
            return selected

        # Phase 1b: check if the message matches any baseline tool's tags.
        # If so, the task clearly needs only baseline tools — skip the LLM call.
        baseline_matched = any(
            any(tag in text for tag in t.get("tags", [])) for t in baseline
        )
        if baseline_matched:
            names = [t["function"]["name"] for t in all_tools]
            self.logger.tool(
                "router phase-1 → %s (baseline tag match, skipped phase-2)", names
            )
            return all_tools

        # Phase 2: LLM fallback — only reached when zero tag matches found anywhere.
        all_names = [t["function"]["name"] for t in all_tools]
        routing_prompt = (
            "You are a tool router. Given the message below, return a JSON "
            "array of tool names "
            f"from this list that are relevant: {all_names}.\n"
            "Return only a JSON array, nothing else. "
            "Return [] if no tools are needed.\n\n"
            f"Message: {message}"
        )
        try:
            resp = self.chat_completion(
                [{"role": "user", "content": routing_prompt}],
                tools=[],
                kind="tool_router",
            )
            content = extract_message(resp).get("content", "") or ""
            # Try multiple extraction strategies in order of reliability.
            routed_names = None
            for attempt in [
                lambda c: json.loads(c.strip()),
                lambda c: json.loads(_search_bracketed(c)),
                lambda c: json.loads(re.sub(r"```(?:json)?\s*|\s*```", "", c).strip()),
            ]:
                try:
                    routed_names = attempt(content)
                    break
                except Exception:
                    continue
            if routed_names is None:
                raise ValueError(
                    f"could not extract JSON array from: {content[:120]!r}"
                )
        except Exception as exc:
            self.logger.warning("router phase-2 failed (%s), using all tools", exc)
            routed_names = all_names

        # Keep baseline + whatever the LLM selected; filter to valid names only
        valid = {t["function"]["name"] for t in all_tools}
        routed_names = [n for n in routed_names if n in valid]
        selected_names = set(routed_names) | _BASELINE_TOOLS
        selected = [t for t in all_tools if t["function"]["name"] in selected_names]
        self.logger.tool(
            "router phase-2 → %s", [t["function"]["name"] for t in selected]
        )
        return selected

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

    def _validate_tool_output(
        self, result: str, func_name: str, is_summary: bool = False
    ) -> str:
        if not result.strip():
            return "(tool returned empty output)"
        if not is_summary and len(result) > MAX_TOOL_OUTPUT_CHARS:
            log_hint = (
                f" Use read_file(file_path='{self._session_log_path}') "
                "to inspect the full output."
                if self._session_log_path
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
        prompt = (
            "/no_think\n"
            f"Summarize the following `{func_name}` tool output concisely "
            "for an AI coding agent. "
            "Include: exit status, key findings, any errors, and what the "
            "agent needs to know to continue its task. "
            "Be factual and brief — 3 to 8 sentences max.\n\n"
            f"Output:\n{json.dumps(artifact, indent=2)[:12_000]}"
        )
        try:
            resp = self.chat_completion(
                [{"role": "user", "content": prompt}],
                tools=[],
                kind="tool_output_summarizer",
                max_tokens=512,
            )
            summary = extract_message(resp).get("content") or ""
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
            f"\n[Full output stored as artifact — use "
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
        for wrong, correct in self._ARG_ALIASES.get(func_name, {}).items():
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
        result = self._before_hook(func_name, args)
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
            modified = self._after_hook(func_name, args, result)
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
    ) -> dict | None:
        """Ask the LLM to fix wrong args after a TypeError.

        Returns corrected args, or None on failure.
        """
        correction_messages = list(getattr(self, "messages", [])) + [
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
        correction_resp = self.chat_completion(
            correction_messages,
            tools=getattr(self, "tools", None),
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
    ) -> tuple[str, str, dict]:
        """Dispatch func_name with arg-correction retries.

        Returns (result, status, record).
        """
        if not hasattr(registry, func_name):
            self.logger.error("not found: '%s' is not registered", func_name)
            valid = [n for n in dir(ToolsRegistry) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            return (
                err,
                "error",
                {
                    "name": func_name,
                    "args": args,
                    "ok": False,
                    "error": f"tool '{func_name}' not found",
                    "tool_call_id": tool_call_id,
                    "artifact": False,
                    "truncated": False,
                    "status": "error",
                },
            )

        sig = inspect.signature(getattr(ToolsRegistry, func_name))

        for attempt in range(self._MAX_ARG_RETRIES + 1):
            try:
                result, is_artifact, is_truncated = self._dispatch_tool(
                    registry, func_name, args, tool_call_id
                )
                self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
                return (
                    result,
                    "success",
                    {
                        "name": func_name,
                        "args": args,
                        "ok": True,
                        "tool_call_id": tool_call_id,
                        "artifact": is_artifact,
                        "truncated": is_truncated,
                        "status": "success",
                    },
                )

            except (
                TypeError
            ) as wrong_args_exc:  # wrong arguments — attempt correction with the LLM
                self.logger.error(
                    "wrong args %s → %s (attempt %s/%s): %s",
                    tool_call_id,
                    func_name,
                    attempt + 1,
                    self._MAX_ARG_RETRIES,
                    wrong_args_exc,
                )
                retries_exhausted = attempt == self._MAX_ARG_RETRIES
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
                    )
                )
                if corrected_args is None:
                    err = (
                        f"Error: wrong arguments for '{func_name}' after "
                        f"{attempt + 1} attempt(s): {wrong_args_exc}. "
                        f"Expected: {func_name}{sig}"
                    )
                    return (
                        err,
                        "error",
                        {
                            "name": func_name,
                            "args": args,
                            "ok": False,
                            "error": str(wrong_args_exc),
                            "tool_call_id": tool_call_id,
                            "artifact": False,
                            "truncated": False,
                            "status": "error",
                        },
                    )
                args = corrected_args

            # other errors — log and return as an error result (don't re-raise,
            # so the agent can keep going)
            except Exception as exc:
                if not isinstance(exc, self._RECOVERABLE_EXCEPTIONS):
                    self.logger.error(
                        "non-recoverable error %s → %s: %s",
                        tool_call_id,
                        func_name,
                        exc,
                    )
                    raise
                self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
                err = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                return (
                    err,
                    "error",
                    {
                        "name": func_name,
                        "args": args,
                        "ok": False,
                        "error": str(exc),
                        "tool_call_id": tool_call_id,
                        "artifact": False,
                        "truncated": False,
                        "status": "error",
                    },
                )

        # Unreachable: the final iteration always returns (success, or an error result
        # once retries are exhausted). Present so mypy can prove the function returns.
        raise AssertionError("invoke_tool retry loop exited without returning")

    def execute_tool_calls(
        self, message: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Dispatch all tool calls in message, returning (tool_messages, call_records).

        Success record: {"name": str, "args": dict, "ok": True}
        Failure record: {"name": str, "args": dict, "ok": False, "error": str}
        """
        tool_calls = message.get("tool_calls", []) or []
        messages, records = [], []
        registry = ToolsRegistry(
            artifacts=self.tool_artifacts, tools=getattr(self, "tools", [])
        )
        self.logger.tool("dispatch: %d tool call(s)", len(tool_calls))

        for tool_call in tool_calls:
            # Parse and validate the raw tool call first, to catch issues
            # before invoking any tools.
            tool_call_id, func_name, args, error = self.parse_tool_call(tool_call)
            if error:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": error,
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "(tool call skipped)",
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
            result, status, record = self.invoke_tool(
                tool_call_id, func_name, args, registry, tool_call
            )

            # Post-process the result before sending it back to the LLM.
            result = self.after_tool_call(tool_call_id, func_name, args, result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                    "status": status,
                }
            )
            records.append(record)

        return messages, records

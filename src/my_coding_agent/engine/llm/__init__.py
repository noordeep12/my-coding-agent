"""LLM HTTP client for an OpenAI-compatible chat-completions server.

Defines ``LLM``, the pure client that owns the ``httpx`` session, calls the
OpenAI-compatible ``/chat/completions`` endpoint with retries, records token
usage per call, and resolves the model's context window lazily. Tool routing and
tool execution live in their own collaborators (``routing.ToolRouter`` and
``tool_execution.ToolExecutor``), which hold an ``LLM`` as their client.
"""

import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from httpx import Response

from ...utils import get_logger
from ...utils.exceptions import APIResponseError

load_dotenv()
OMLX_API_URL = os.environ.get("OMLX_API_URL", "http://127.0.0.1:8321/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "changeme")
OMLX_MODEL = os.environ.get("OMLX_MODEL", "Qwen3.6-35B-A3B-4bit")


# Default per-request HTTP timeout (seconds) for the LLM API session.
DEFAULT_HTTP_TIMEOUT = 30.0
# Number of attempts for transient (connection/timeout) failures on external calls.
_HTTP_RETRIES = 3
# Base backoff (seconds) between retries; doubles each attempt.
_HTTP_BACKOFF = 0.5
# Errors worth retrying — transient connectivity/timeout, not protocol errors.
_TRANSIENT_HTTP_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)


class LLM:
    """Pure HTTP client for an OpenAI-compatible LLM server.

    Own the ``httpx`` session and issue chat-completion requests, recording every
    call's token usage in ``self.llm_calls`` tagged by ``kind``. Construction
    performs no network I/O: the model's context window is probed lazily on first
    access to ``context_window``. Tool routing and tool execution are provided by
    the ``routing.ToolRouter`` and ``tool_execution.ToolExecutor`` collaborators,
    which hold an ``LLM`` instance as their client.
    """

    # 128k fallback used when the server is unreachable or omits a context length.
    DEFAULT_CONTEXT_WINDOW = 131_072

    def __init__(
        self,
        api_url: str = OMLX_API_URL,
        api_key: str = OMLX_API_KEY,
        model: str = OMLX_MODEL,
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
        # Optional observability recorder; set by Agent. None → no capture.
        self._recorder: Any = None
        self.llm_calls: list[dict] = []  # one entry per chat_completion call, in order

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
        _t0 = time.monotonic()
        resp = self._request_with_retry(
            "POST",
            self.api_url + "/chat/completions",
            json=body,
        )
        _latency = time.monotonic() - _t0
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
        # Observability capture (separate from logging). The single choke point
        # for every call kind: records latency, tokens, the input conversation
        # snapshot, the tool definitions given this turn, and the response. No-op
        # when no recorder is attached.
        if self._recorder is not None:
            self._recorder.record_llm_call(
                kind=kind,
                call=call_num,
                latency_s=_latency,
                usage=usage,
                messages=messages,
                context_window=self.context_window,
                response_data=data,
                tools=tools,
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

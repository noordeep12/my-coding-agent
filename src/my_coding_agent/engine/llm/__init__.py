"""LLM HTTP client for an OpenAI-compatible chat-completions server.

Defines ``LLM``, the pure client that owns the ``httpx`` session, calls the
OpenAI-compatible ``/chat/completions`` endpoint with retries, records token
usage per call, and resolves the model's context window lazily. Tool execution
lives in its own collaborator (``tool_execution.ToolExecutor``), which holds an
``LLM`` as its client.
"""

import json
import os
import time
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from httpx import Response

from ...utils import get_logger
from .errors import LLMHTTPStatusError, LLMMalformedBodyError, LLMTransportError
from .schema import (
    DEFAULT_OUTAGE_TOLERANCE_S,
    OUTAGE_TOLERANCE_ENV,
    PATIENT_BACKOFF_CAP_S,
)

load_dotenv()
OMLX_API_URL = os.environ.get("OMLX_API_URL", "http://127.0.0.1:8321/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "changeme")
OMLX_MODEL = os.environ.get("OMLX_MODEL", "Qwen3.6-35B-A3B-6bit")


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
    access to ``context_window``. Tool execution is provided by the
    ``tool_execution.ToolExecutor`` collaborator, which holds an ``LLM`` instance
    as its client.
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

    @staticmethod
    def _outage_tolerance_s() -> float:
        """Patient-phase ceiling (seconds) from ``MCA_LLM_OUTAGE_TOLERANCE_S``.

        Falls back to ``DEFAULT_OUTAGE_TOLERANCE_S`` when the env var is unset or
        unparseable, so a bad value never crashes a run — it just uses the default.
        """
        raw = os.environ.get(OUTAGE_TOLERANCE_ENV)
        if raw is None:
            return DEFAULT_OUTAGE_TOLERANCE_S
        try:
            return float(raw)
        except ValueError:
            return DEFAULT_OUTAGE_TOLERANCE_S

    @staticmethod
    def _validate_response(resp: Response) -> dict:
        """Classify a chat-completion response, returning parsed JSON on success.

        Raises a classified error (never returns a failed response as success):
        - ``LLMHTTPStatusError`` for any non-2xx status (retryable for 5xx/429);
        - ``LLMMalformedBodyError`` for a non-JSON body, or JSON with a
          missing/empty ``choices`` list (an empty turn is a failure, not a
          silent success).
        """
        status = resp.status_code
        if not 200 <= status < 300:
            retryable = status == 429 or 500 <= status < 600
            raise LLMHTTPStatusError(
                f"LLM API returned HTTP {status}. Body prefix: {resp.text[:200]!r}",
                status_code=status,
                retryable=retryable,
                hint="A 5xx/429 is transient (server restarting or overloaded); "
                "other 4xx indicate a malformed request or auth problem.",
            )
        try:
            data: dict = resp.json()
        except Exception as exc:
            raise LLMMalformedBodyError(
                f"API returned non-JSON response (HTTP {status}): {exc}. "
                f"Body prefix: {resp.text[:200]!r}",
                hint="Check that api_url points at an OpenAI-compatible endpoint "
                "and the server is healthy.",
            ) from exc
        if not data.get("choices"):
            raise LLMMalformedBodyError(
                f"API returned HTTP {status} with missing/empty 'choices'. "
                f"Body prefix: {resp.text[:200]!r}",
                hint="The server accepted the request but produced no completion; "
                "it may have crashed or swapped models mid-response.",
            )
        return data

    def _post_chat_with_recovery(
        self, body: dict, kind: str, call_num: int
    ) -> tuple[Response, dict]:
        """POST one chat completion, absorbing a transient outage (D2).

        Two-phase recovery: the fast transport retries inside
        ``_request_with_retry`` run per probe, then — for retryable
        classifications (transport, HTTP 5xx/429) — this loop keeps probing with
        capped exponential backoff until the server answers or the outage
        tolerance (``MCA_LLM_OUTAGE_TOLERANCE_S``) is exhausted. Non-retryable
        classifications (malformed body, other 4xx) fail immediately without
        consuming the window. Returns the validated response and its parsed JSON.

        Raises:
            LLMTransportError | LLMHTTPStatusError | LLMMalformedBodyError:
                The classified failure when unrecoverable (tolerance exceeded, or
                a non-retryable class).
        """
        tolerance = self._outage_tolerance_s()
        url = self.api_url + "/chat/completions"
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            err: Exception
            classification: str
            try:
                resp = self._request_with_retry("POST", url, json=body)
                data = self._validate_response(resp)
                if attempt > 1:
                    stalled = time.monotonic() - start
                    self.logger.warning(
                        "LLM recovered after %.1fs stall (%d attempts) "
                        "[call #%d, kind=%s]",
                        stalled,
                        attempt,
                        call_num,
                        kind,
                    )
                    if self._recorder is not None:
                        self._recorder.record_llm_recovery(
                            kind=kind,
                            call=call_num,
                            attempts=attempt,
                            stalled_s=round(stalled, 3),
                        )
                return resp, data
            except _TRANSIENT_HTTP_ERRORS as exc:
                classification = LLMTransportError.classification
                err = LLMTransportError(str(exc))
            except (LLMHTTPStatusError, LLMMalformedBodyError) as exc:
                classification = exc.classification
                err = exc
                if not exc.retryable:
                    self._record_failure(
                        kind,
                        call_num,
                        classification,
                        attempt,
                        time.monotonic() - start,
                    )
                    raise

            elapsed = time.monotonic() - start
            if elapsed >= tolerance:
                self.logger.error(
                    "LLM outage exceeded tolerance %.0fs (%s) — giving up after "
                    "%d attempts [call #%d, kind=%s]",
                    tolerance,
                    classification,
                    attempt,
                    call_num,
                    kind,
                )
                self._record_failure(kind, call_num, classification, attempt, elapsed)
                raise err
            delay = min(PATIENT_BACKOFF_CAP_S, _HTTP_BACKOFF * (2 ** (attempt - 1)))
            delay = min(delay, tolerance - elapsed)  # never oversleep the deadline
            self.logger.warning(
                "LLM %s failure (attempt %d) — waiting %.1fs before retry "
                "(%.0fs/%.0fs tolerance used) [call #%d, kind=%s]",
                classification,
                attempt,
                delay,
                elapsed,
                tolerance,
                call_num,
                kind,
            )
            if self._recorder is not None:
                self._recorder.record_llm_wait(
                    kind=kind,
                    call=call_num,
                    classification=classification,
                    attempt=attempt,
                    delay_s=round(delay, 3),
                    elapsed_s=round(elapsed, 3),
                )
            time.sleep(delay)

    def _record_failure(
        self,
        kind: str,
        call_num: int,
        classification: str,
        attempt: int,
        elapsed: float,
    ) -> None:
        """Emit an ``llm_failure`` observation when a recorder is attached."""
        if self._recorder is not None:
            self._recorder.record_llm_failure(
                kind=kind,
                call=call_num,
                classification=classification,
                attempts=attempt,
                elapsed_s=round(elapsed, 3),
            )

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
                ``tool_router``, ``tool_arg_correction``).
            max_tokens: Optional cap on generated tokens; omitted when None.

        Returns:
            The raw ``httpx.Response`` from the completions endpoint.

        Raises:
            LLMMalformedBodyError: Non-JSON body, or JSON with missing/empty
                ``choices`` (never surfaced as an empty assistant turn).
            LLMHTTPStatusError: Non-2xx status; unrecoverable after the patient
                retry phase for 5xx/429, or immediately for other 4xx.
            LLMTransportError: Connection/timeout that outlasts the outage
                tolerance (``MCA_LLM_OUTAGE_TOLERANCE_S``).
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
        _started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        _t0 = time.monotonic()
        resp, data = self._post_chat_with_recovery(body, kind, call_num)
        _latency = time.monotonic() - _t0
        self.logger.api(
            "← %d (%d bytes)  [call #%d, kind=%s]",
            resp.status_code,
            len(resp.content),
            call_num,
            kind,
        )
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
                started_at=_started_at,
                max_tokens=max_tokens,
                resources=self._recorder.resource_window(_t0, time.monotonic()),
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

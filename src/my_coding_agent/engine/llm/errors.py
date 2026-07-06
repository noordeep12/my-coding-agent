"""Classified chat-completion failures (run-resilience D1).

Every failure of ``LLM.chat_completion`` is raised as exactly one of these,
each carrying a ``classification`` (one of the ``CLASSIFICATION_*`` constants)
and a ``retryable`` flag the resilience loop uses to decide fast+patient retry
versus fail-fast. A failed response must never become a successful completion
or an empty assistant turn — the single choke point raises one of these first.
"""

from __future__ import annotations

from ...utils.exceptions import APIResponseError, MyCodingAgentError
from .schema import (
    CLASSIFICATION_HTTP_STATUS,
    CLASSIFICATION_MALFORMED_BODY,
    CLASSIFICATION_TRANSPORT,
)


class LLMCallError(MyCodingAgentError):
    """Base for a classified chat-completion failure.

    ``classification`` is one of the ``CLASSIFICATION_*`` constants; ``retryable``
    tells the resilience loop whether the fast+patient retry phase applies.
    """

    classification: str = ""
    retryable: bool = False


class LLMTransportError(LLMCallError):
    """Connection/timeout failure reaching the server — always retryable."""

    classification = CLASSIFICATION_TRANSPORT
    retryable = True


class LLMHTTPStatusError(LLMCallError):
    """The server answered with a non-2xx status.

    5xx and 429 are transient (server restarting / overloaded) and retryable;
    other 4xx are protocol bugs and fail fast.
    """

    classification = CLASSIFICATION_HTTP_STATUS

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retryable: bool,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.status_code = status_code
        self.retryable = retryable


class LLMMalformedBodyError(LLMCallError, APIResponseError):
    """A 2xx response whose body is non-JSON, or JSON with missing/empty choices.

    Subclasses ``APIResponseError`` (a ``ValueError``) so existing callers that
    catch the pre-classification error still catch it. Never retryable — a
    healthy server does not return a malformed body transiently.
    """

    classification = CLASSIFICATION_MALFORMED_BODY
    retryable = False

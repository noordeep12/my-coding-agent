"""Shared fixtures for the test suite.

These fixtures build lightweight, network-free objects so unit tests stay fast,
deterministic, and independent — never hitting a real LLM server or network.
"""

import logging

import pytest

from my_coding_agent.llm import LLM
from my_coding_agent.routing import ToolRouter
from my_coding_agent.tool_execution import ToolExecutor


class _FakeResponse:
    """Minimal stand-in for an httpx.Response carrying a fixed JSON payload."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_response():
    """Return a factory that wraps any payload as a JSON-returning response."""
    return _FakeResponse


@pytest.fixture
def silent_logger():
    """A logger with the custom level methods used across the package, no output."""
    logger = logging.getLogger("test-silent")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    # The package uses custom level methods (tool/api/llm); stub them as no-ops.
    for name in ("tool", "api", "llm"):
        if not hasattr(logger, name):
            setattr(logger, name, lambda *a, **k: None)
    return logger


@pytest.fixture
def bare_llm(silent_logger):
    """An LLM instance with a silent logger and no network I/O at construction.

    ``LLM.__init__`` is network-free (the context-window probe is deferred to
    first access of ``context_window``), so we construct normally and only swap
    in the silent logger. Tests that exercise ``available_models`` patch
    ``_request_with_retry`` so the deferred probe never reaches a real server.
    """
    llm = LLM()
    llm.logger = silent_logger
    return llm


@pytest.fixture
def bare_router(bare_llm, silent_logger):
    """A ToolRouter wrapping a bare LLM client, both with silent loggers.

    Network-free: phase-2 routing calls ``client.chat_completion``, which tests
    patch directly on ``bare_llm`` so no request reaches a real server.
    """
    router = ToolRouter(bare_llm)
    router.logger = silent_logger
    return router


@pytest.fixture
def bare_executor(bare_llm, silent_logger):
    """A ToolExecutor wrapping a bare LLM client, both with silent loggers.

    Network-free: the argument-correction and artifact-summarization calls go
    through ``client.chat_completion``, which tests patch on ``bare_llm``. The
    executor owns ``tool_artifacts`` (execution state).
    """
    executor = ToolExecutor(bare_llm)
    executor.logger = silent_logger
    return executor

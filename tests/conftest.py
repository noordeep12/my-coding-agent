"""Shared fixtures for the test suite.

These fixtures build lightweight, network-free objects so unit tests stay fast,
deterministic, and independent — never hitting a real LLM server or network.
"""

import logging

import pytest

from my_coding_agent.llm import LLM


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
    """An LLM instance built WITHOUT __init__ — no network call to /models.

    __init__ performs an HTTP GET to discover the context window, which we must
    not trigger in unit tests. We allocate the object directly and attach only
    the attributes the pure methods under test actually read.
    """
    llm = object.__new__(LLM)
    llm.logger = silent_logger
    llm.llm_calls = []
    llm.tool_artifacts = {}
    return llm

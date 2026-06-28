"""Shared utilities: exceptions, parsing, logging, and terminal UI."""

import logging

from .logging_core import (
    API,
    LLM,
    TOOL,
    ColoredFormatter,
    DynamicStderrHandler,
    attach_session_log,
    detach_session_log,
    get_logger,
)
from .terminal_ui import print_banner, print_run_summary

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "get_logger",
    "attach_session_log",
    "detach_session_log",
    "print_banner",
    "print_run_summary",
    "ColoredFormatter",
    "DynamicStderrHandler",
    "TOOL",
    "API",
    "LLM",
]

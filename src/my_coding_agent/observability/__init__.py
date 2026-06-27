# ruff: noqa: F401
"""Observability for agent sessions: logging, terminal UI, and event capture."""

import sys  # re-exported so tests may patch ``observability.sys``

from .logging_core import (
    _ANSI_RE,
    _CUSTOM_LEVELS,
    _LEVEL_COLORS,
    API,
    LLM,
    TOOL,
    ColoredFormatter,
    DynamicStderrHandler,
    _get_package_logger,
    _PackageLogger,
    _register_level_names,
    _SessionLogHandle,
    _TeeStream,
    attach_session_log,
    detach_session_log,
    get_logger,
)
from .recorder import Recorder, current_recorder, current_session_id
from .terminal_ui import (
    _context_resets_section,
    _git_branch,
    _handoff_rows,
    _markdown_rows,
    _SummaryStyle,
    _token_chart_rows,
    _tool_call_rows,
    _tool_calls_section,
    _tool_count_label,
    print_banner,
    print_run_summary,
)

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
    "Recorder",
    "current_session_id",
    "current_recorder",
]

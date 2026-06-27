"""Logging, session-log capture, and terminal-UI rendering for the agent.

Two focused modules (CONTRIBUTE.md §5, §16):

- ``logging_core`` — custom TOOL/API/LLM levels, ``ColoredFormatter``,
  ``DynamicStderrHandler``, ``get_logger``, and the ``_TeeStream`` /
  ``attach_session_log`` / ``detach_session_log`` stderr-capture helpers.
- ``terminal_ui`` — the box-drawn ``print_banner`` and ``print_run_summary``
  renderers, the shared ``_git_branch`` helper, and all row/section/chart
  sub-helpers.
"""

# ruff: noqa: F401

import sys  # re-exported so tests may patch ``logger.sys``

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
]

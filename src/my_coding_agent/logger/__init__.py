"""Logging, session-log capture, and terminal-UI rendering for the agent.

This package splits three independent concerns into focused modules (CONTRIBUTE.md
§5, §16) while preserving the historical flat import surface — code may keep using
``from my_coding_agent.logger import get_logger, print_banner, ...``:

- ``logging_core`` — custom TOOL/API/LLM levels, ``ColoredFormatter``,
  ``DynamicStderrHandler``, and the ``get_logger`` factory.
- ``session_log`` — the ``_TeeStream`` and ``attach_session_log`` /
  ``detach_session_log`` helpers that fan stderr to plain + colored log files.
- ``banner`` — the box-drawn ``print_banner`` renderer plus the shared
  ``_git_branch`` helper.
- ``summary`` — the box-drawn ``print_run_summary`` renderer and its
  row/section/chart sub-helpers.
"""

# This package is a re-export facade: every import below is a deliberate
# re-export of a submodule symbol kept on the historical flat surface, so F401
# (imported-but-unused) does not apply to this file.
# ruff: noqa: F401

import sys  # re-exported so tests may patch ``logger.sys``

from .banner import (
    _git_branch,
    print_banner,
)
from .logging_core import (
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
    get_logger,
)
from .session_log import (
    _ANSI_RE,
    _SessionLogHandle,
    _TeeStream,
    attach_session_log,
    detach_session_log,
)
from .summary import (
    _context_resets_section,
    _handoff_rows,
    _markdown_rows,
    _SummaryStyle,
    _token_chart_rows,
    _tool_call_rows,
    _tool_calls_section,
    _tool_count_label,
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

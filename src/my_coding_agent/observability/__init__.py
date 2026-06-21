"""Local observability for agent sessions: capture layer + reader.

The :class:`Recorder` captures a structured ``events.jsonl`` per session while a
run executes (LLM calls, tool I/O, handoffs, agent/subagent links); :func:`reader`
helpers parse that stream into typed :class:`Session` objects with analytical
views. The HTML viewer (``report.py``) renders them. The ``logger`` package is
untouched — capture is a separate concern.
"""

from .events import Handoff, LLMCall, Session, ToolCall, TreeNode
from .reader import load_all_sessions, load_session
from .recorder import Recorder, current_recorder, current_session_id
from .report import build_payload, render_html, write_report
from .tree import build_trace_tree

__all__ = [
    "Recorder",
    "current_session_id",
    "current_recorder",
    "load_session",
    "load_all_sessions",
    "build_trace_tree",
    "build_payload",
    "render_html",
    "write_report",
    "Session",
    "LLMCall",
    "ToolCall",
    "Handoff",
    "TreeNode",
]

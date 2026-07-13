"""Trace Explorer — read-side viewer for per-session events.jsonl streams.

Public surface:

- ``TraceNode``     — one node in the visualised trace graph
- ``TraceSession``  — full parsed session with nodes, edges, and analytics
- ``load_session``  — parse an ``events.jsonl`` file into a ``TraceSession``
- ``list_sessions`` — scan a ``.my_coding_agent/`` directory for all sessions
"""

import logging

from .reader import list_sessions, load_session
from .schema import TraceNode, TraceSession

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "TraceNode",
    "TraceSession",
    "list_sessions",
    "load_session",
]

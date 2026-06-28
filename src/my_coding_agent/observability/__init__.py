"""Observability — passive event capture for agent sessions."""

from .recorder import Recorder, current_recorder, current_session_id

__all__ = [
    "Recorder",
    "current_session_id",
    "current_recorder",
]

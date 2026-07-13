"""Observability — passive event capture for agent sessions."""

from .error_classification import classify_error
from .recorder import Recorder, current_recorder, current_session_id, now

__all__ = [
    "Recorder",
    "current_session_id",
    "current_recorder",
    "classify_error",
    "now",
]

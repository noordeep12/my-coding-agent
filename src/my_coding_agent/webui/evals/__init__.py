"""Evaluations serving surface: read-side view models, embedded page, CRUD/run API."""

import logging

from .api import handle_evaluation_route
from .reader import (
    CaseView,
    RunSummary,
    RunView,
    list_available_datasets,
    list_runs,
    load_run,
)
from .server import eval_dashboard_html, handle_eval_api_route

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CaseView",
    "RunSummary",
    "RunView",
    "list_runs",
    "load_run",
    "list_available_datasets",
    "eval_dashboard_html",
    "handle_eval_api_route",
    "handle_evaluation_route",
]

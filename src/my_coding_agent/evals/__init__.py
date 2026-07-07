"""Evals — repeatable case runner and result store for the agent."""

from .cases import load_case_set
from .results import EvalRunResult, build_run_result, load_run_result, write_run_result
from .runner import run_case_set
from .schema import EvalCase, EvalScore
from .scoring import Scorer, UnknownScorerError, register_scorer, resolve_scorer

__all__ = [
    "EvalCase",
    "EvalScore",
    "EvalRunResult",
    "Scorer",
    "UnknownScorerError",
    "load_case_set",
    "run_case_set",
    "build_run_result",
    "write_run_result",
    "load_run_result",
    "register_scorer",
    "resolve_scorer",
]

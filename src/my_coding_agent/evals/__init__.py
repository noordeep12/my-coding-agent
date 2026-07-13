"""Evals — declarative run configs, scorers, and a two-run comparison gate."""

from . import judge  # noqa: F401  (import registers the "judge" scorer)
from .compare import (
    CaseFlip,
    ComparisonResult,
    DatasetVersionMismatchError,
    ThresholdConfig,
    Verdict,
    compare_runs,
    evaluate_verdict,
)
from .results import EvalRunResult, build_run_result, load_run_result, write_run_result
from .schema import EvalCase, EvalScore
from .scoring import Scorer, UnknownScorerError, register_scorer, resolve_scorer

# Importing registers the "trajectory" scorer (#140) in the registry above.
from .trajectory import TrajectoryScorer

__all__ = [
    "EvalCase",
    "EvalScore",
    "EvalRunResult",
    "Scorer",
    "TrajectoryScorer",
    "UnknownScorerError",
    "build_run_result",
    "write_run_result",
    "load_run_result",
    "register_scorer",
    "resolve_scorer",
    "judge",
    "CaseFlip",
    "ComparisonResult",
    "DatasetVersionMismatchError",
    "ThresholdConfig",
    "Verdict",
    "compare_runs",
    "evaluate_verdict",
]

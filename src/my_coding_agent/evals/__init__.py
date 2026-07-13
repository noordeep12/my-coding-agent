"""Evals — declarative run configs, scorers, and a two-run comparison gate."""

from . import judge
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
    "CaseFlip",
    "ComparisonResult",
    "DatasetVersionMismatchError",
    "EvalCase",
    "EvalRunResult",
    "EvalScore",
    "Scorer",
    "ThresholdConfig",
    "TrajectoryScorer",
    "UnknownScorerError",
    "Verdict",
    "build_run_result",
    "compare_runs",
    "evaluate_verdict",
    "judge",
    "load_run_result",
    "register_scorer",
    "resolve_scorer",
    "write_run_result",
]

"""Evals — repeatable case runner, result store, and versioned datasets."""

from .cases import load_case_set
from .datasets import (
    Dataset,
    add_case,
    add_failure_case,
    create_dataset,
    dataset_ref,
    list_datasets,
    load_dataset,
    resolve_cases,
    retire_case,
    run_dataset,
)
from .results import EvalRunResult, build_run_result, load_run_result, write_run_result
from .runner import run_case_set
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
    "Dataset",
    "load_case_set",
    "run_case_set",
    "build_run_result",
    "write_run_result",
    "load_run_result",
    "register_scorer",
    "resolve_scorer",
    "create_dataset",
    "load_dataset",
    "add_case",
    "retire_case",
    "add_failure_case",
    "list_datasets",
    "resolve_cases",
    "run_dataset",
    "dataset_ref",
]

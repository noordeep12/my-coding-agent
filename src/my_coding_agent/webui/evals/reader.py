"""Read-only view models over the persisted eval result store.

Mirrors ``viewer/reader.py``'s shape: pure functions that turn on-disk
records into typed, JSON-serialisable view models for the eval dashboard.
Never runs or mutates an eval — it only reads what `evals.results` and
`evals.datasets` already wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...evals.cases import load_case_set
from ...evals.datasets import DEFAULT_CASES_DIR, DEFAULT_DATASETS_DIR, Dataset
from ...evals.datasets import list_datasets as _list_datasets
from ...evals.results import RESULTS_ROOT, EvalRunResult, load_run_result

# A run whose `pass_rate` reaches this fraction of cases passing is a "pass"
# verdict — mirrors `evals/cli.py`'s own exit-code threshold.
_PASS_THRESHOLD = 1.0


@dataclass(frozen=True)
class CaseView:
    """One case's score, joined with its task/expected when the case still
    exists on disk (absent for a case retired or renamed since the run)."""

    case_id: str
    passed: bool
    metrics: dict[str, float]
    detail: dict[str, Any]
    task: str | None = None
    expected: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunSummary:
    """One run's identity and headline verdict, for run history / overview."""

    run_id: str
    timestamp: str
    dataset: str
    model: str
    verdict: str  # "pass" | "fail"
    headline_score: float | None
    case_count: int


@dataclass(frozen=True)
class RunView:
    """A single run's full breakdown: summary, aggregates, and per-case detail."""

    summary: RunSummary
    aggregate_metrics: dict[str, float]
    cases: list[CaseView] = field(default_factory=list)


def _verdict(aggregate_metrics: dict[str, float]) -> str:
    return "pass" if aggregate_metrics.get("pass_rate") == _PASS_THRESHOLD else "fail"


def _summarize(result: EvalRunResult) -> RunSummary:
    return RunSummary(
        run_id=result.run_id,
        timestamp=result.timestamp,
        dataset=result.dataset,
        model=result.model,
        verdict=_verdict(result.aggregate_metrics),
        headline_score=result.aggregate_metrics.get("pass_rate"),
        case_count=len(result.scores),
    )


def list_runs(root: Path = RESULTS_ROOT) -> list[RunSummary]:
    """Enumerate available eval runs, most recent first.

    A run directory with a missing or corrupt ``result.json`` is skipped
    rather than failing the whole listing (older/partial records degrade to
    absent, not an error).

    Args:
        root: Directory holding one subdirectory per run (`evals/results.py`'s
            ``RESULTS_ROOT`` by default).

    Returns:
        One `RunSummary` per readable run, sorted by timestamp descending.
    """
    if not root.exists():
        return []
    summaries = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not (entry / "result.json").exists():
            continue
        try:
            result = load_run_result(entry)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        summaries.append(_summarize(result))
    summaries.sort(key=lambda s: s.timestamp, reverse=True)
    return summaries


def load_run(
    run_id: str,
    *,
    root: Path = RESULTS_ROOT,
    cases_dir: Path = DEFAULT_CASES_DIR,
) -> RunView | None:
    """Load one run's full breakdown, joined against the current case set.

    Args:
        run_id: The run to load.
        root: Directory holding one subdirectory per run.
        cases_dir: Directory eval cases load from, for joining task/expected
            onto each case's score.

    Returns:
        The run's view, or `None` if the run id has no readable record.
    """
    run_dir = root / run_id
    if not (run_dir / "result.json").exists():
        return None
    try:
        result = load_run_result(run_dir)
    except (OSError, ValueError, KeyError, TypeError):
        return None

    by_id = {case.id: case for case in load_case_set(cases_dir)}
    cases = [
        CaseView(
            case_id=score.case_id,
            passed=score.passed,
            metrics=score.metrics,
            detail=score.detail,
            task=by_id[score.case_id].task if score.case_id in by_id else None,
            expected=by_id[score.case_id].expected if score.case_id in by_id else None,
        )
        for score in result.scores
    ]
    return RunView(
        summary=_summarize(result),
        aggregate_metrics=result.aggregate_metrics,
        cases=cases,
    )


def list_available_datasets(*, base_dir: Path = DEFAULT_DATASETS_DIR) -> list[Dataset]:
    """Enumerate available datasets and their current versions.

    Thin pass-through to `evals.datasets.list_datasets` so the viewer layer
    has a single import surface (`evals_reader`) mirroring `reader.py`'s
    `list_sessions`.
    """
    return _list_datasets(base_dir=base_dir)

"""Compare two eval run results and turn the comparison into a verdict.

`EvalRunResult.dataset` carries the exact `id@vVERSION` stamp
(`datasets.dataset_ref`); comparisons use it to refuse a metric move that
resulted from a changed test set rather than a real change, and to report
both per-metric aggregate deltas and which individual cases flipped
pass<->fail — since a flat aggregate can hide a subset regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .results import EvalRunResult


class DatasetVersionMismatchError(Exception):
    """Raised when two runs did not execute the same dataset id+version."""

    def __init__(self, baseline_dataset: str, candidate_dataset: str) -> None:
        super().__init__(
            f"Runs are not comparable: baseline used dataset "
            f"'{baseline_dataset}', candidate used '{candidate_dataset}'"
        )
        self.baseline_dataset = baseline_dataset
        self.candidate_dataset = candidate_dataset


@dataclass(frozen=True)
class CaseFlip:
    """A case whose pass/fail verdict changed between the two runs.

    Args:
        case_id: The `EvalCase.id` that flipped.
        baseline_passed: Whether the case passed in the baseline run.
        candidate_passed: Whether the case passed in the candidate run.
    """

    case_id: str
    baseline_passed: bool
    candidate_passed: bool

    @property
    def is_regression(self) -> bool:
        """True if the case went from passing to failing."""
        return self.baseline_passed and not self.candidate_passed


@dataclass(frozen=True)
class ComparisonResult:
    """The full comparison between a baseline and a candidate run.

    Args:
        baseline_run_id: The baseline `EvalRunResult.run_id`.
        candidate_run_id: The candidate `EvalRunResult.run_id`.
        dataset: The shared `id@vVERSION` dataset stamp both runs used.
        metric_deltas: Per-metric `candidate - baseline` deltas.
        flips: Cases whose pass/fail verdict changed, in dataset order.
    """

    baseline_run_id: str
    candidate_run_id: str
    dataset: str
    metric_deltas: dict[str, float] = field(default_factory=dict)
    flips: list[CaseFlip] = field(default_factory=list)

    @property
    def regressions(self) -> list[CaseFlip]:
        """Flips that went from passing to failing."""
        return [flip for flip in self.flips if flip.is_regression]


def compare_runs(
    baseline: EvalRunResult,
    candidate: EvalRunResult,
    *,
    allow_version_mismatch: bool = False,
) -> ComparisonResult:
    """Compare two run results at aggregate and per-case granularity.

    Args:
        baseline: The prior run to compare against.
        candidate: The new run being evaluated.
        allow_version_mismatch: If False (default), raise
            `DatasetVersionMismatchError` when the two runs used different
            dataset id+version stamps. If True, proceed anyway (the caller
            is responsible for surfacing a loud warning).

    Returns:
        The full comparison: metric deltas and the per-case flip set.

    Raises:
        DatasetVersionMismatchError: If the runs used different dataset
            stamps and `allow_version_mismatch` is False.
    """
    if baseline.dataset != candidate.dataset and not allow_version_mismatch:
        raise DatasetVersionMismatchError(baseline.dataset, candidate.dataset)

    all_metrics = set(baseline.aggregate_metrics) | set(candidate.aggregate_metrics)
    metric_deltas = {
        metric: candidate.aggregate_metrics.get(metric, 0.0)
        - baseline.aggregate_metrics.get(metric, 0.0)
        for metric in sorted(all_metrics)
    }

    baseline_by_case = {score.case_id: score.passed for score in baseline.scores}
    candidate_by_case = {score.case_id: score.passed for score in candidate.scores}
    shared_case_ids = [
        case_id for case_id in baseline_by_case if case_id in candidate_by_case
    ]
    flips = [
        CaseFlip(
            case_id=case_id,
            baseline_passed=baseline_by_case[case_id],
            candidate_passed=candidate_by_case[case_id],
        )
        for case_id in shared_case_ids
        if baseline_by_case[case_id] != candidate_by_case[case_id]
    ]

    return ComparisonResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        dataset=candidate.dataset,
        metric_deltas=metric_deltas,
        flips=flips,
    )


@dataclass(frozen=True)
class ThresholdConfig:
    """Configurable pass/fail thresholds applied to a comparison.

    Args:
        metric_floors: Per-metric minimum value the candidate run must
            meet or exceed (not the delta — the candidate's raw value).
        no_case_regressed: If True (default), any pass->fail flip fails
            the verdict.
    """

    metric_floors: dict[str, float] = field(default_factory=dict)
    no_case_regressed: bool = True


@dataclass(frozen=True)
class Verdict:
    """The pass/fail outcome of applying thresholds to a comparison.

    Args:
        passed: Overall verdict.
        reasons: Human-readable reasons for failure; empty when passed.
    """

    passed: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_verdict(
    comparison: ComparisonResult,
    candidate: EvalRunResult,
    thresholds: ThresholdConfig,
) -> Verdict:
    """Turn a comparison into a pass/fail verdict against `thresholds`.

    Args:
        comparison: The comparison to evaluate.
        candidate: The candidate run, whose raw metric values are checked
            against `thresholds.metric_floors`.
        thresholds: The configured floors and regression rule.

    Returns:
        The verdict: passed, plus a reason for every violated floor and
        every regressed case.
    """
    reasons: list[str] = []

    for metric, floor in sorted(thresholds.metric_floors.items()):
        value = candidate.aggregate_metrics.get(metric, 0.0)
        if value < floor:
            reasons.append(f"metric '{metric}' = {value} is below floor {floor}")

    if thresholds.no_case_regressed:
        for flip in comparison.regressions:
            reasons.append(f"case '{flip.case_id}' regressed (pass -> fail)")

    return Verdict(passed=not reasons, reasons=reasons)

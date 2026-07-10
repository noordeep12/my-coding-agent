"""Tests for evals/compare.py — run comparison, guard, and threshold verdict."""

from __future__ import annotations

import pytest

from my_coding_agent.evals.compare import (
    DatasetVersionMismatchError,
    ThresholdConfig,
    compare_runs,
    evaluate_verdict,
)
from my_coding_agent.evals.results import build_run_result
from my_coding_agent.evals.schema import EvalScore


def _result(dataset, scores, aggregate_metrics):
    return build_run_result(
        dataset=dataset, scores=scores, aggregate_metrics=aggregate_metrics
    )


def test_compare_reports_metric_deltas_and_flip_set():
    baseline = _result(
        "smoke@v1",
        [
            EvalScore("a", True, {}, {}),
            EvalScore("b", True, {}, {}),
            EvalScore("c", False, {}, {}),
        ],
        {"pass_rate": 0.667},
    )
    candidate = _result(
        "smoke@v1",
        [
            EvalScore("a", True, {}, {}),
            EvalScore("b", False, {}, {}),
            EvalScore("c", True, {}, {}),
        ],
        {"pass_rate": 0.667},
    )

    comparison = compare_runs(baseline, candidate)

    assert comparison.metric_deltas == {"pass_rate": pytest.approx(0.0)}
    flips = {flip.case_id: flip for flip in comparison.flips}
    assert set(flips) == {"b", "c"}
    assert flips["b"].is_regression
    assert not flips["c"].is_regression


def test_subset_regression_visible_under_flat_aggregate():
    baseline = _result(
        "smoke@v1",
        [EvalScore("a", True, {}, {}), EvalScore("b", False, {}, {})],
        {"pass_rate": 0.5},
    )
    candidate = _result(
        "smoke@v1",
        [EvalScore("a", False, {}, {}), EvalScore("b", True, {}, {})],
        {"pass_rate": 0.5},
    )

    comparison = compare_runs(baseline, candidate)

    assert comparison.metric_deltas["pass_rate"] == pytest.approx(0.0)
    assert len(comparison.regressions) == 1
    assert comparison.regressions[0].case_id == "a"


def test_same_dataset_version_compares_cleanly():
    baseline = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})
    candidate = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})

    comparison = compare_runs(baseline, candidate)

    assert comparison.dataset == "smoke@v1"
    assert comparison.flips == []


def test_different_dataset_version_is_refused_by_default():
    baseline = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})
    candidate = _result("smoke@v2", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})

    with pytest.raises(DatasetVersionMismatchError):
        compare_runs(baseline, candidate)


def test_different_dataset_version_allowed_with_override():
    baseline = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})
    candidate = _result("smoke@v2", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})

    comparison = compare_runs(baseline, candidate, allow_version_mismatch=True)

    assert comparison.dataset == "smoke@v2"


def test_metric_floor_violation_fails_verdict():
    baseline = _result(
        "smoke@v1",
        [EvalScore("a", True, {}, {})],
        {"precision": 0.7, "recall": 0.9},
    )
    candidate = _result(
        "smoke@v1",
        [EvalScore("a", True, {}, {})],
        {"precision": 0.9, "recall": 0.6},
    )
    comparison = compare_runs(baseline, candidate)
    thresholds = ThresholdConfig(metric_floors={"recall": 0.8})

    verdict = evaluate_verdict(comparison, candidate, thresholds)

    assert not verdict.passed
    assert any("recall" in reason for reason in verdict.reasons)


def test_strict_improvement_passes():
    baseline = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 0.5})
    candidate = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 0.9})
    comparison = compare_runs(baseline, candidate)
    thresholds = ThresholdConfig(metric_floors={"pass_rate": 0.8})

    verdict = evaluate_verdict(comparison, candidate, thresholds)

    assert verdict.passed
    assert verdict.reasons == []


def test_single_regressed_case_fails_no_regression_rule():
    baseline = _result(
        "smoke@v1",
        [EvalScore("a", True, {}, {}), EvalScore("b", True, {}, {})],
        {"pass_rate": 1.0},
    )
    candidate = _result(
        "smoke@v1",
        [EvalScore("a", True, {}, {}), EvalScore("b", False, {}, {})],
        {"pass_rate": 0.5},
    )
    comparison = compare_runs(baseline, candidate)
    thresholds = ThresholdConfig()

    verdict = evaluate_verdict(comparison, candidate, thresholds)

    assert not verdict.passed
    assert any("b" in reason for reason in verdict.reasons)


def test_no_regression_rule_can_be_disabled():
    baseline = _result("smoke@v1", [EvalScore("a", True, {}, {})], {"pass_rate": 1.0})
    candidate = _result("smoke@v1", [EvalScore("a", False, {}, {})], {"pass_rate": 0.0})
    comparison = compare_runs(baseline, candidate)
    thresholds = ThresholdConfig(no_case_regressed=False)

    verdict = evaluate_verdict(comparison, candidate, thresholds)

    assert verdict.passed

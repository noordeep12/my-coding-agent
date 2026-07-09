"""Tests for evals/datasets.py: dataset model, lifecycle, and lister."""

from __future__ import annotations

import json

import pytest

from my_coding_agent.evals.datasets import (
    CaseNotInDatasetError,
    Dataset,
    DatasetError,
    DatasetNotFoundError,
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
from my_coding_agent.evals.schema import EvalScore


def test_create_and_load_dataset_reports_identity_and_cases(tmp_path):
    base_dir = tmp_path / "datasets"
    created = create_dataset("smoke", ["case-1", "case-2"], base_dir=base_dir)

    assert created == Dataset(id="smoke", version=1, case_ids=("case-1", "case-2"))

    loaded = load_dataset("smoke", base_dir=base_dir)
    assert loaded == created


def test_create_dataset_twice_raises(tmp_path):
    base_dir = tmp_path / "datasets"
    create_dataset("dup", ["case-1"], base_dir=base_dir)

    with pytest.raises(DatasetError):
        create_dataset("dup", ["case-2"], base_dir=base_dir)


def test_load_missing_dataset_raises(tmp_path):
    base_dir = tmp_path / "datasets"
    with pytest.raises(DatasetNotFoundError):
        load_dataset("nope", base_dir=base_dir)


def test_add_case_bumps_version_and_preserves_prior_membership(tmp_path):
    base_dir = tmp_path / "datasets"
    create_dataset("ds", ["case-1"], base_dir=base_dir)

    updated = add_case("ds", "case-2", base_dir=base_dir)

    assert updated.version == 2
    assert updated.case_ids == ("case-1", "case-2")

    prior = load_dataset("ds", version=1, base_dir=base_dir)
    assert prior.case_ids == ("case-1",)

    latest = load_dataset("ds", base_dir=base_dir)
    assert latest == updated


def test_retire_case_bumps_version_and_preserves_prior_membership(tmp_path):
    base_dir = tmp_path / "datasets"
    create_dataset("ds", ["case-1", "case-2"], base_dir=base_dir)

    updated = retire_case("ds", "case-1", base_dir=base_dir)

    assert updated.version == 2
    assert updated.case_ids == ("case-2",)

    prior = load_dataset("ds", version=1, base_dir=base_dir)
    assert prior.case_ids == ("case-1", "case-2")


def test_retire_unknown_case_raises(tmp_path):
    base_dir = tmp_path / "datasets"
    create_dataset("ds", ["case-1"], base_dir=base_dir)

    with pytest.raises(CaseNotInDatasetError):
        retire_case("ds", "case-missing", base_dir=base_dir)


def test_add_failure_case_writes_case_file_and_bumps_dataset(tmp_path):
    base_dir = tmp_path / "datasets"
    cases_dir = tmp_path / "cases"
    create_dataset("ds", [], base_dir=base_dir)

    updated = add_failure_case(
        "ds",
        case_id="regression-1",
        task="Do the thing that failed",
        scorer="exact_match",
        expected={"output": "correct"},
        base_dir=base_dir,
        cases_dir=cases_dir,
    )

    assert updated.version == 2
    assert updated.case_ids == ("regression-1",)

    case_file = cases_dir / "regression-1.json"
    assert case_file.exists()
    payload = json.loads(case_file.read_text())
    assert payload["id"] == "regression-1"
    assert payload["task"] == "Do the thing that failed"
    assert payload["scorer"] == "exact_match"
    assert payload["expected"] == {"output": "correct"}


def test_add_failure_case_rejects_duplicate_case_id(tmp_path):
    base_dir = tmp_path / "datasets"
    cases_dir = tmp_path / "cases"
    create_dataset("ds", [], base_dir=base_dir)
    add_failure_case(
        "ds",
        case_id="dup",
        task="t",
        scorer="exact_match",
        expected={},
        base_dir=base_dir,
        cases_dir=cases_dir,
    )

    with pytest.raises(DatasetError):
        add_failure_case(
            "ds",
            case_id="dup",
            task="t2",
            scorer="exact_match",
            expected={},
            base_dir=base_dir,
            cases_dir=cases_dir,
        )


def test_list_datasets_reports_each_at_its_current_version(tmp_path):
    base_dir = tmp_path / "datasets"
    create_dataset("alpha", ["a1"], base_dir=base_dir)
    create_dataset("beta", ["b1"], base_dir=base_dir)
    add_case("beta", "b2", base_dir=base_dir)

    datasets = list_datasets(base_dir=base_dir)

    assert datasets == [
        Dataset(id="alpha", version=1, case_ids=("a1",)),
        Dataset(id="beta", version=2, case_ids=("b1", "b2")),
    ]


def test_list_datasets_empty_when_dir_missing(tmp_path):
    assert list_datasets(base_dir=tmp_path / "nonexistent") == []


def _write_case(cases_dir, case_id, **overrides):
    cases_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": case_id,
        "task": "do the thing",
        "scorer": "exact_match",
        "expected": {"contains": "ok"},
    }
    payload.update(overrides)
    (cases_dir / f"{case_id}.json").write_text(json.dumps(payload))


def test_dataset_ref_formats_id_and_version():
    assert dataset_ref(Dataset(id="smoke", version=3, case_ids=())) == "smoke@v3"


def test_resolve_cases_loads_in_dataset_order(tmp_path):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case-a")
    _write_case(cases_dir, "case-b")
    dataset = Dataset(id="ds", version=1, case_ids=("case-b", "case-a"))

    cases = resolve_cases(dataset, cases_dir=cases_dir)

    assert [c.id for c in cases] == ["case-b", "case-a"]


def test_resolve_cases_skips_missing_case_id(tmp_path, caplog):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case-a")
    dataset = Dataset(id="ds", version=1, case_ids=("case-a", "case-missing"))

    cases = resolve_cases(dataset, cases_dir=cases_dir)

    assert [c.id for c in cases] == ["case-a"]


def test_run_dataset_stamps_dataset_id_and_version(tmp_path, mocker):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case-a")
    results_root = tmp_path / "results"
    dataset = Dataset(id="ds", version=2, case_ids=("case-a",))

    fake_score = EvalScore(case_id="case-a", passed=True, metrics={}, detail={})
    mocker.patch(
        "my_coding_agent.evals.datasets.run_case_set",
        return_value=([fake_score], {"pass_rate": 1.0}),
    )

    result = run_dataset(dataset, cases_dir=cases_dir, results_root=results_root)

    assert result.dataset == "ds@v2"
    assert result.scores == [fake_score]
    assert (results_root / result.run_id / "result.json").exists()


def test_run_dataset_bumped_version_stamps_new_ref(tmp_path, mocker):
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case-a")
    results_root = tmp_path / "results"

    mocker.patch(
        "my_coding_agent.evals.datasets.run_case_set",
        return_value=([], {"pass_rate": 0.0}),
    )

    first = run_dataset(
        Dataset(id="ds", version=1, case_ids=("case-a",)),
        cases_dir=cases_dir,
        results_root=results_root,
    )
    second = run_dataset(
        Dataset(id="ds", version=2, case_ids=("case-a",)),
        cases_dir=cases_dir,
        results_root=results_root,
    )

    assert first.dataset == "ds@v1"
    assert second.dataset == "ds@v2"

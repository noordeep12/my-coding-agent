"""Tests for evals/datasets.py: dataset model, lifecycle, and lister."""

from __future__ import annotations

import pytest

from my_coding_agent.evals.datasets import (
    CaseNotInDatasetError,
    Dataset,
    DatasetError,
    DatasetNotFoundError,
    add_case,
    add_failure_case,
    create_dataset,
    list_datasets,
    load_dataset,
    retire_case,
)


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
        scorer_ref="exact_match",
        expected={"output": "correct"},
        base_dir=base_dir,
        cases_dir=cases_dir,
    )

    assert updated.version == 2
    assert updated.case_ids == ("regression-1",)

    case_file = cases_dir / "regression-1.json"
    assert case_file.exists()
    import json

    payload = json.loads(case_file.read_text())
    assert payload["id"] == "regression-1"
    assert payload["task"] == "Do the thing that failed"
    assert payload["scorer_ref"] == "exact_match"
    assert payload["expected"] == {"output": "correct"}


def test_add_failure_case_rejects_duplicate_case_id(tmp_path):
    base_dir = tmp_path / "datasets"
    cases_dir = tmp_path / "cases"
    create_dataset("ds", [], base_dir=base_dir)
    add_failure_case(
        "ds",
        case_id="dup",
        task="t",
        scorer_ref="exact_match",
        expected={},
        base_dir=base_dir,
        cases_dir=cases_dir,
    )

    with pytest.raises(DatasetError):
        add_failure_case(
            "ds",
            case_id="dup",
            task="t2",
            scorer_ref="exact_match",
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

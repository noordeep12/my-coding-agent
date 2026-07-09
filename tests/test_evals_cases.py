"""Tests for evals/cases.py — the eval case loader."""

from __future__ import annotations

import json

from my_coding_agent.evals.cases import load_case_set


def _write_case(path, **fields):
    path.write_text(json.dumps(fields))


def test_well_formed_cases_load(tmp_path):
    _write_case(
        tmp_path / "a.json",
        id="a",
        task="do a",
        scorer="exact_match",
        expected={"equals": "x"},
    )
    _write_case(
        tmp_path / "b.json",
        id="b",
        task="do b",
        scorer="exact_match",
        expected={"contains": "y"},
        dataset="ds",
        tags=["smoke"],
    )

    cases = load_case_set(tmp_path)

    assert [case.id for case in cases] == ["a", "b"]
    assert cases[1].dataset == "ds"
    assert cases[1].tags == ("smoke",)


def test_malformed_case_is_skipped_not_fatal(tmp_path, caplog):
    (tmp_path / "bad.json").write_text("{not json")
    _write_case(
        tmp_path / "good.json",
        id="good",
        task="do it",
        scorer="exact_match",
        expected={"equals": "x"},
    )

    cases = load_case_set(tmp_path)

    assert [case.id for case in cases] == ["good"]


def test_missing_required_key_is_skipped(tmp_path):
    _write_case(tmp_path / "missing.json", id="m", task="do it")

    cases = load_case_set(tmp_path)

    assert cases == []


def test_duplicate_id_is_skipped_keeping_first(tmp_path):
    _write_case(
        tmp_path / "1_first.json",
        id="dup",
        task="first",
        scorer="exact_match",
        expected={"equals": "x"},
    )
    _write_case(
        tmp_path / "2_second.json",
        id="dup",
        task="second",
        scorer="exact_match",
        expected={"equals": "y"},
    )

    cases = load_case_set(tmp_path)

    assert len(cases) == 1
    assert cases[0].task == "first"


def test_empty_dir_loads_no_cases(tmp_path):
    assert load_case_set(tmp_path) == []

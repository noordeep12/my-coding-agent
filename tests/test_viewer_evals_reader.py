"""Tests for viewer/evals_reader.py — read-only eval-result view models."""

from __future__ import annotations

import json

from my_coding_agent.evals.results import build_run_result, write_run_result
from my_coding_agent.evals.schema import EvalScore
from my_coding_agent.viewer.evals_reader import list_runs, load_run


def _scores():
    return [
        EvalScore(
            case_id="c1",
            passed=True,
            metrics={"match": 1.0},
            detail={"reason": "equals check", "expected": "hi", "actual": "hi"},
        ),
        EvalScore(
            case_id="c2",
            passed=False,
            metrics={"match": 0.0},
            detail={"reason": "equals check", "expected": "bye", "actual": "no"},
        ),
    ]


def _write_case_file(cases_dir, case_id, task, expected):
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "id": case_id,
                "task": task,
                "scorer": "exact_match",
                "expected": expected,
            }
        )
    )


def test_list_runs_returns_newest_first(tmp_path):
    older = build_run_result(
        dataset="ex@v1", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )
    write_run_result(older, root=tmp_path)

    newer = build_run_result(
        dataset="ex@v1", scores=_scores(), aggregate_metrics={"pass_rate": 1.0}
    )
    # Force a distinguishable, later timestamp than `older`.
    import dataclasses

    newer = dataclasses.replace(newer, timestamp="2999-01-01T00:00:00+00:00")
    write_run_result(newer, root=tmp_path)

    summaries = list_runs(root=tmp_path)

    assert [s.run_id for s in summaries] == [newer.run_id, older.run_id]
    assert summaries[0].verdict == "pass"  # pass_rate reaches the threshold
    assert summaries[0].headline_score == 1.0
    assert summaries[0].case_count == 2
    assert summaries[1].verdict == "fail"


def test_list_runs_empty_root_returns_empty_list(tmp_path):
    assert list_runs(root=tmp_path / "does-not-exist") == []


def test_list_runs_skips_a_corrupt_record(tmp_path):
    good = build_run_result(
        dataset="ex@v1", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )
    write_run_result(good, root=tmp_path)

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "result.json").write_text("{not valid json")

    summaries = list_runs(root=tmp_path)

    assert [s.run_id for s in summaries] == [good.run_id]


def test_load_run_joins_case_task_and_expected(tmp_path):
    cases_dir = tmp_path / "cases"
    _write_case_file(cases_dir, "c1", "say hi", {"equals": "hi"})
    _write_case_file(cases_dir, "c2", "say bye", {"equals": "bye"})

    result = build_run_result(
        dataset="ex@v1", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )
    write_run_result(result, root=tmp_path)

    view = load_run(result.run_id, root=tmp_path, cases_dir=cases_dir)

    assert view is not None
    assert view.summary.run_id == result.run_id
    assert view.summary.verdict == "fail"
    failed = next(c for c in view.cases if c.case_id == "c2")
    assert failed.task == "say bye"
    assert failed.expected == {"equals": "bye"}
    assert failed.detail["actual"] == "no"


def test_load_run_degrades_gracefully_when_case_file_is_gone(tmp_path):
    result = build_run_result(
        dataset="ex@v1", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )
    write_run_result(result, root=tmp_path)

    view = load_run(result.run_id, root=tmp_path, cases_dir=tmp_path / "no-cases")

    assert view is not None
    assert all(c.task is None and c.expected is None for c in view.cases)


def test_load_run_missing_run_id_returns_none(tmp_path):
    assert load_run("does-not-exist", root=tmp_path) is None

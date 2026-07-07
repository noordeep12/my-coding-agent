"""Tests for evals/results.py — the versioned result record and its writer."""

from __future__ import annotations

from my_coding_agent.evals.results import (
    build_run_result,
    load_run_result,
    write_run_result,
)
from my_coding_agent.evals.schema import RESULT_SCHEMA_VERSION, EvalScore


def _scores():
    return [
        EvalScore(case_id="c1", passed=True, metrics={"match": 1.0}, detail={}),
        EvalScore(case_id="c2", passed=False, metrics={"match": 0.0}, detail={}),
    ]


def test_completed_run_writes_a_readable_record(tmp_path):
    result = build_run_result(
        dataset="example", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )

    run_dir = write_run_result(result, root=tmp_path)

    assert (run_dir / "result.json").exists()
    loaded = load_run_result(run_dir)
    assert loaded.run_id == result.run_id


def test_record_round_trips_full_identity(tmp_path):
    result = build_run_result(
        dataset="example", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )

    run_dir = write_run_result(result, root=tmp_path)
    loaded = load_run_result(run_dir)

    assert loaded.schema_version == RESULT_SCHEMA_VERSION
    assert loaded.dataset == "example"
    assert loaded.agent_version == result.agent_version
    assert loaded.model == result.model
    assert loaded.timestamp == result.timestamp
    assert [s.case_id for s in loaded.scores] == ["c1", "c2"]
    assert loaded.aggregate_metrics == {"pass_rate": 0.5}


def test_older_reader_tolerates_a_newer_record_with_unknown_keys(tmp_path):
    result = build_run_result(dataset="example", scores=_scores(), aggregate_metrics={})
    run_dir = write_run_result(result, root=tmp_path)

    raw = (run_dir / "result.json").read_text()
    patched = raw.replace(
        '"schema_version": 1', '"schema_version": 1, "future_field": 42'
    )
    (run_dir / "result.json").write_text(patched)

    loaded = load_run_result(run_dir)

    assert loaded.run_id == result.run_id

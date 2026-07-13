"""Tests for evals/results.py — the versioned result record and its writer."""

from __future__ import annotations

import json

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


def test_score_without_session_id_loads_as_none(tmp_path):
    result = build_run_result(
        dataset="example", scores=_scores(), aggregate_metrics={"pass_rate": 0.5}
    )

    run_dir = write_run_result(result, root=tmp_path)
    loaded = load_run_result(run_dir)

    assert all(score.session_id is None for score in loaded.scores)


def test_pre_existing_record_without_session_id_field_still_loads(tmp_path):
    result = build_run_result(dataset="example", scores=_scores(), aggregate_metrics={})
    run_dir = write_run_result(result, root=tmp_path)

    raw = json.loads((run_dir / "result.json").read_text())
    for score in raw["scores"]:
        del score["session_id"]
    (run_dir / "result.json").write_text(json.dumps(raw))

    loaded = load_run_result(run_dir)

    assert [s.case_id for s in loaded.scores] == ["c1", "c2"]
    assert all(s.session_id is None for s in loaded.scores)


def test_real_session_gets_a_verdict_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_id = "sess-123"
    session_dir = tmp_path / ".my_coding_agent" / session_id
    session_dir.mkdir(parents=True)

    scores = [
        EvalScore(
            case_id="c1",
            passed=True,
            metrics={"match": 1.0},
            detail={"reason": "ok"},
            session_id=session_id,
        )
    ]
    result = build_run_result(
        dataset="example", scores=scores, aggregate_metrics={"pass_rate": 1.0}
    )
    result_dir = write_run_result(result)

    verdict = json.loads((session_dir / "verdict.json").read_text())
    assert verdict["run_id"] == result.run_id
    assert verdict["case_id"] == "c1"
    assert verdict["passed"] is True
    assert verdict["result_path"] == str(result_dir / "result.json")


def test_temp_workspace_session_gets_no_verdict_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scores = [
        EvalScore(
            case_id="c1",
            passed=True,
            metrics={},
            detail={},
            session_id="a-session-that-never-existed",
        )
    ]
    result = build_run_result(
        dataset="example", scores=scores, aggregate_metrics={"pass_rate": 1.0}
    )

    write_run_result(result)

    assert not (tmp_path / ".my_coding_agent" / "a-session-that-never-existed").exists()


def test_verdict_artifact_write_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_id = "sess-456"
    session_dir = tmp_path / ".my_coding_agent" / session_id
    session_dir.mkdir(parents=True)
    # A directory at the target path forces write_text to raise IsADirectoryError.
    (session_dir / "verdict.json").mkdir()

    scores = [
        EvalScore(case_id="c1", passed=True, metrics={}, detail={}, session_id=session_id)
    ]
    result = build_run_result(
        dataset="example", scores=scores, aggregate_metrics={"pass_rate": 1.0}
    )

    run_dir = write_run_result(result)

    assert (run_dir / "result.json").exists()


def test_trace_files_are_untouched_by_verdict_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_id = "sess-789"
    session_dir = tmp_path / ".my_coding_agent" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text("existing-trace\n")
    (session_dir / "session_data.json").write_text('{"existing": true}')

    scores = [
        EvalScore(case_id="c1", passed=True, metrics={}, detail={}, session_id=session_id)
    ]
    result = build_run_result(
        dataset="example", scores=scores, aggregate_metrics={"pass_rate": 1.0}
    )
    write_run_result(result)

    assert (session_dir / "events.jsonl").read_text() == "existing-trace\n"
    assert (session_dir / "session_data.json").read_text() == '{"existing": true}'

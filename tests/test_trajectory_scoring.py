"""Tests for the deterministic trajectory scorer (issue #140)."""

from __future__ import annotations

import json

from my_coding_agent.evals.schema import EvalCase
from my_coding_agent.evals.scoring import RunResult, resolve_scorer
from my_coding_agent.evals.trajectory import (
    TRAJECTORY_SCORER_REF,
    CostRollup,
    EfficiencyBaseline,
    TrajectoryScorer,
    load_trajectory,
    score_argument_validity,
    score_efficiency,
    score_error_handling,
    score_redundancy,
    score_tool_selection,
)
from my_coding_agent.observability.recorder import Recorder


def _make_session(tmp_path, name="sess"):
    """Build a real session dir with events.jsonl (via Recorder) + session_data.json."""
    sdir = tmp_path / name
    sdir.mkdir()
    rec = Recorder(session_id=name, session_dir=sdir)
    rec.start(
        label="agent", model="test-model", context_window=8192, posture="sandboxed"
    )
    return rec, sdir


_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_RESPONSE = {"choices": [{"message": {"content": "ok"}}]}


def _record_step_boundary(rec, call, messages):
    """Emit a main llm_call — the step boundary ``_group_into_steps`` requires."""
    rec.record_llm_call(
        kind="main",
        call=call,
        latency_s=0.1,
        usage=_USAGE,
        messages=messages,
        context_window=8192,
        response_data=_RESPONSE,
    )


def _record_ok(rec, name, args, result="ok"):
    rec.before_tool(name, args)
    rec.after_tool(name, args, result=result, ok=True, error=None)


def _record_fail(rec, name, args, error):
    rec.before_tool(name, args)
    rec.after_tool(name, args, result="", ok=False, error=error)


def _finish_and_write_session_data(rec, sdir, tool_records, steps=1, elapsed_s=1.0):
    rec.finish(stop_reason="stop", steps=steps, elapsed_s=elapsed_s)
    data = {
        "session_id": sdir.name,
        "steps": steps,
        "elapsed_seconds": elapsed_s,
        "total_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        "tool_records": tool_records,
    }
    (sdir / "session_data.json").write_text(json.dumps(data))


class TestLoadTrajectory:
    def test_loads_tool_sequence_and_cost_figures_from_fixture_session(self, tmp_path):
        rec, sdir = _make_session(tmp_path)
        _record_step_boundary(rec, 1, [{"role": "user", "content": "do it"}])
        _record_ok(rec, "read_file", {"file_path": "a.py"})
        _record_ok(rec, "write_file", {"file_path": "a.py", "content": "x"})
        tool_records = [
            {
                "name": "read_file",
                "args": {"file_path": "a.py"},
                "ok": True,
                "status": "success",
            },
            {
                "name": "write_file",
                "args": {"file_path": "a.py", "content": "x"},
                "ok": True,
                "status": "success",
            },
        ]
        _finish_and_write_session_data(rec, sdir, tool_records, steps=2, elapsed_s=3.5)

        trace = load_trajectory(sdir)

        assert [r["name"] for r in trace.tool_records] == ["read_file", "write_file"]
        assert trace.cost.steps == 2
        assert trace.cost.elapsed_s == 3.5
        assert trace.cost.total_tokens == 150
        tool_nodes = [n for n in trace.session.nodes.values() if n.type == "tool_call"]
        assert len(tool_nodes) == 2

    def test_missing_session_data_yields_empty_cost_and_records(self, tmp_path):
        rec, sdir = _make_session(tmp_path)
        rec.finish(stop_reason="stop", steps=0, elapsed_s=0.0)

        trace = load_trajectory(sdir)

        assert trace.tool_records == []
        assert trace.cost == CostRollup(0, 0, 0, 0, 0.0, None)


class TestToolSelection:
    def test_clean_path_scores_full_marks(self):
        records = [
            {"name": "read_file", "ok": True},
            {"name": "write_file", "ok": True},
        ]
        score = score_tool_selection(records)
        assert score.value == 1.0
        assert score.detail["off_toolset_calls"] == []

    def test_not_found_call_is_flagged_and_lowers_score(self):
        records = [
            {"name": "read_file", "ok": True},
            {
                "name": "frobnicate",
                "ok": False,
                "error": "Error: tool 'frobnicate' not found. Available tools: []",
            },
        ]
        score = score_tool_selection(records)
        assert score.value == 0.5
        assert score.detail["off_toolset_calls"] == [1]

    def test_empty_run_scores_full_marks(self):
        assert score_tool_selection([]).value == 1.0


class TestArgumentValidity:
    def test_wrong_args_call_is_flagged_and_lowers_score(self):
        records = [
            {"name": "read_file", "ok": True},
            {
                "name": "write_file",
                "ok": False,
                "error": (
                    "Error: wrong arguments for 'write_file': missing content. "
                    "Expected: write_file(file_path, content)"
                ),
            },
        ]
        score = score_argument_validity(records)
        assert score.value == 0.5
        assert score.detail["wrong_args_calls"] == [1]

    def test_not_found_failure_does_not_count_as_wrong_args(self):
        records = [
            {
                "name": "frobnicate",
                "ok": False,
                "error": "Error: tool 'frobnicate' not found. Available tools: []",
            },
        ]
        score = score_argument_validity(records)
        assert score.value == 1.0
        assert score.detail["wrong_args_calls"] == []


class TestErrorHandling:
    def test_clean_path_scores_full_marks(self):
        records = [{"name": "read_file", "ok": True}] * 3
        score = score_error_handling(records)
        assert score.value == 1.0
        assert score.detail["ignored_failure_streaks"] == []

    def test_ignored_repeated_failure_lowers_score(self):
        failing = {
            "name": "bash",
            "ok": False,
            "error": "Error: tool 'bash' raised OSError: boom",
        }
        records = [failing, failing, failing, {"name": "read_file", "ok": True}]
        score = score_error_handling(records)
        assert score.value < 1.0
        assert len(score.detail["ignored_failure_streaks"]) == 1
        assert score.detail["ignored_failure_streaks"][0]["length"] == 3

    def test_single_failure_below_threshold_does_not_flag(self):
        records = [
            {
                "name": "bash",
                "ok": False,
                "error": "Error: tool 'bash' raised OSError: x",
            }
        ]
        score = score_error_handling(records)
        assert score.value == 1.0
        assert score.detail["ignored_failure_streaks"] == []


class TestRedundancy:
    def test_lean_run_has_no_trailing_work(self):
        records = [{"name": "read_file"}, {"name": "write_file"}]
        score = score_redundancy(records)
        assert score.value == 1.0
        assert score.detail["trailing_calls_after_last_mutation"] == 0

    def test_verification_spiral_scores_lower_than_lean_run(self):
        lean = [{"name": "read_file"}, {"name": "write_file"}]
        spiral = [
            {"name": "read_file"},
            {"name": "write_file"},
            {"name": "read_file"},
            {"name": "read_file"},
            {"name": "read_tool_artifact"},
        ]

        lean_score = score_redundancy(lean)
        spiral_score = score_redundancy(spiral)

        assert spiral_score.value < lean_score.value
        assert spiral_score.detail["trailing_calls_after_last_mutation"] == 3

    def test_no_mutation_at_all_counts_every_call_as_trailing(self):
        records = [{"name": "read_file"}, {"name": "read_tool_artifact"}]
        score = score_redundancy(records)
        assert score.detail["trailing_calls_after_last_mutation"] == 2
        assert score.value == 0.0


class TestEfficiency:
    def test_without_baseline_only_redundancy_scores(self):
        cost = CostRollup(
            steps=10,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            elapsed_s=5.0,
        )
        records = [{"name": "write_file"}, {"name": "read_file"}]
        score = score_efficiency(cost, records)
        assert score.value == score_redundancy(records).value
        assert score.detail["steps"] == 10

    def test_baseline_within_limits_scores_full_marks(self):
        cost = CostRollup(
            steps=2,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            elapsed_s=1.0,
        )
        records = [{"name": "write_file"}]
        baseline = EfficiencyBaseline(
            max_steps=5, max_total_tokens=100, max_elapsed_s=10.0
        )
        score = score_efficiency(cost, records, baseline=baseline)
        assert score.value == 1.0

    def test_spiral_fixture_scores_lower_than_lean_fixture_on_efficiency(self):
        lean_cost = CostRollup(
            steps=2,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            elapsed_s=1.0,
        )
        lean_records = [{"name": "write_file"}, {"name": "read_file"}]

        spiral_cost = CostRollup(
            steps=20,
            prompt_tokens=500,
            completion_tokens=300,
            total_tokens=800,
            elapsed_s=30.0,
        )
        spiral_records = [{"name": "write_file"}] + [{"name": "read_file"}] * 10

        baseline = EfficiencyBaseline(
            max_steps=5, max_total_tokens=100, max_elapsed_s=10.0
        )

        lean_score = score_efficiency(lean_cost, lean_records, baseline=baseline)
        spiral_score = score_efficiency(spiral_cost, spiral_records, baseline=baseline)

        assert spiral_score.value < lean_score.value


class TestTrajectoryScorer:
    def _case(self, **expected):
        return EvalCase(
            id="c1", task="do it", scorer=TRAJECTORY_SCORER_REF, expected=expected
        )

    def test_registers_under_trajectory_ref(self):
        assert isinstance(resolve_scorer(TRAJECTORY_SCORER_REF), TrajectoryScorer)

    def test_errored_run_scores_failed_without_reading_the_trace(self, tmp_path):
        run_result = RunResult(
            final_output="",
            session_dir=tmp_path / "does-not-exist",
            session_id="sess1",
            errored=True,
        )
        score = TrajectoryScorer().score(self._case(), run_result)
        assert score.passed is False
        assert score.metrics["trajectory"] == 0.0

    def test_clean_run_produces_a_per_dimension_score_and_passes(self, tmp_path):
        rec, sdir = _make_session(tmp_path)
        _record_step_boundary(rec, 1, [{"role": "user", "content": "do it"}])
        _record_ok(rec, "read_file", {"file_path": "a.py"})
        _record_ok(rec, "write_file", {"file_path": "a.py", "content": "x"})
        tool_records = [
            {"name": "read_file", "args": {}, "ok": True, "status": "success"},
            {"name": "write_file", "args": {}, "ok": True, "status": "success"},
        ]
        _finish_and_write_session_data(rec, sdir, tool_records, steps=2, elapsed_s=1.0)

        run_result = RunResult(
            final_output="done", session_dir=sdir, session_id=sdir.name, errored=False
        )
        score = TrajectoryScorer().score(self._case(), run_result)

        assert score.passed is True
        assert score.metrics["trajectory"] == 1.0
        for dimension in (
            "tool_selection",
            "argument_validity",
            "error_handling",
            "efficiency",
        ):
            assert dimension in score.metrics
            assert dimension in score.detail

    def test_broken_run_scores_lower_than_pass_threshold(self, tmp_path):
        rec, sdir = _make_session(tmp_path)
        _record_step_boundary(rec, 1, [{"role": "user", "content": "do it"}])
        failing_error = "Error: tool 'bash' raised OSError: boom"
        _record_fail(rec, "bash", {}, failing_error)
        _record_fail(rec, "bash", {}, failing_error)
        _record_fail(rec, "bash", {}, failing_error)
        tool_records = [
            {
                "name": "bash",
                "args": {},
                "ok": False,
                "error": failing_error,
                "status": "error",
            }
        ] * 3
        _finish_and_write_session_data(rec, sdir, tool_records, steps=3, elapsed_s=1.0)

        run_result = RunResult(
            final_output="done", session_dir=sdir, session_id=sdir.name, errored=False
        )
        score = TrajectoryScorer().score(self._case(), run_result)

        assert score.metrics["error_handling"] < 1.0
        assert score.metrics["error_handling"] < score.metrics["tool_selection"]

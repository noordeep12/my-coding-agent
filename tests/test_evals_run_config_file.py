"""Tests for evals/run_config_file.py — declarative YAML run config."""

from __future__ import annotations

import json

import pytest
import yaml

from my_coding_agent.engine.agent import AgentNode
from my_coding_agent.engine.llm import LLM
from my_coding_agent.evals import datasets as evals_datasets
from my_coding_agent.evals import run_config_file as rcf
from my_coding_agent.evals.results import load_run_result
from my_coding_agent.viewer import reader as viewer_reader


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _stop_resp(content="pong"):
    return _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


def _write_config(tmp_path, data, name="run.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def _full_config(**overrides):
    data = {
        "llm": {
            "api_url": "http://example.test/v1",
            "model": "test-model",
            "api_key_env": "TEST_API_KEY",
            "timeout": 5,
        },
        "run": {
            "system_prompt": "You are helpful.",
            "task": "say pong",
            "max_steps": 3,
        },
        "evaluation": {
            "checks": [
                {"name": "check1", "evaluator": "exact_match", "expected": "pong"}
            ]
        },
    }
    data.update(overrides)
    return data


# -- loading / validation ----------------------------------------------------


def test_full_config_loads_into_typed_models(tmp_path):
    path = _write_config(tmp_path, _full_config())
    loaded = rcf.load_config_file(path)

    assert loaded.run_config.system_prompt == "You are helpful."
    assert loaded.run_config.user_prompt_template == "say pong"
    assert loaded.run_config.extra_params["max_steps"] == 3
    assert len(loaded.checks) == 1
    assert loaded.checks[0].evaluator == "exact_match"
    assert loaded.content_hash


def test_minimal_config_is_valid_and_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_API_URL", "http://env.test/v1")
    monkeypatch.setenv("OMLX_MODEL", "env-model")
    path = _write_config(
        tmp_path,
        {
            "run": {"task": "say pong"},
            "evaluation": {
                "checks": [{"evaluator": "exact_match", "expected": "pong"}]
            },
        },
    )
    loaded = rcf.load_config_file(path)
    assert loaded.run_config.user_prompt_template == "say pong"

    client = rcf.build_llm_client(loaded.llm_section)
    assert client.api_url == "http://env.test/v1"
    assert client.model == "env-model"


def test_malformed_yaml_is_rejected_with_path(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("llm: [unterminated")

    with pytest.raises(rcf.ConfigValidationError) as excinfo:
        rcf.load_config_file(path)
    assert str(path) in str(excinfo.value)


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = _write_config(tmp_path, {**_full_config(), "bogus": {}})

    with pytest.raises(rcf.ConfigValidationError) as excinfo:
        rcf.load_config_file(path)
    assert "bogus" in str(excinfo.value)


def test_unknown_evaluator_is_rejected(tmp_path):
    data = _full_config()
    data["evaluation"]["checks"][0]["evaluator"] = "not_a_real_scorer"
    path = _write_config(tmp_path, data)

    with pytest.raises(rcf.ConfigValidationError) as excinfo:
        rcf.load_config_file(path)
    assert "not_a_real_scorer" in str(excinfo.value)


def test_raw_api_key_is_rejected(tmp_path):
    data = _full_config()
    data["llm"]["api_key"] = "sk-super-secret"
    path = _write_config(tmp_path, data)

    with pytest.raises(rcf.ConfigValidationError) as excinfo:
        rcf.load_config_file(path)
    assert "api_key_env" in str(excinfo.value)


def test_missing_task_is_rejected(tmp_path):
    data = _full_config()
    del data["run"]["task"]
    path = _write_config(tmp_path, data)

    with pytest.raises(rcf.ConfigValidationError) as excinfo:
        rcf.load_config_file(path)
    assert "run.task" in str(excinfo.value)


def test_evaluation_with_no_criteria_is_rejected(tmp_path):
    data = _full_config()
    data["evaluation"] = {}
    path = _write_config(tmp_path, data)

    with pytest.raises(rcf.ConfigValidationError) as excinfo:
        rcf.load_config_file(path)
    assert "evaluation" in str(excinfo.value)


# -- execution ----------------------------------------------------------


def test_execute_from_config_end_to_end(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_API_KEY", "sekret")

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    path = _write_config(tmp_path, _full_config())
    result, verdict = rcf.execute_from_config(path)

    assert verdict == "pass"
    assert result.config_path == str(path)
    assert result.config_hash
    assert (tmp_path / ".my_coding_agent" / "evals").exists()

    loaded_result = load_run_result(
        tmp_path / ".my_coding_agent" / "evals" / result.run_id
    )
    assert loaded_result.config_path == str(path)
    assert loaded_result.config_hash == result.config_hash


def test_execute_from_config_reflects_failing_verdict(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "nope"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    path = _write_config(tmp_path, _full_config())
    result, verdict = rcf.execute_from_config(path)

    assert verdict == "fail"
    assert result.scores[0].passed is False


def test_execute_from_config_session_carries_standard_trace_files(
    tmp_path, monkeypatch, mocker
):
    monkeypatch.chdir(tmp_path)
    mocker.patch.object(LLM, "chat_completion", return_value=_stop_resp())

    path = _write_config(tmp_path, _full_config())
    result, _ = rcf.execute_from_config(path)

    session_dirs = [
        d for d in (tmp_path / ".my_coding_agent").glob("*") if d.name != "evals"
    ]
    assert session_dirs, "expected an agent session directory under .my_coding_agent"
    session_dir = session_dirs[0]
    assert (session_dir / "events.jsonl").exists()
    assert (session_dir / "session_data.json").exists()
    assert result.scores[0].passed is True

    sessions = viewer_reader.list_sessions(tmp_path / ".my_coding_agent")
    assert any(row["session_id"] == session_dir.name for row in sessions)
    trace = viewer_reader.load_session(session_dir / "events.jsonl")
    assert trace.nodes


def test_execute_from_config_with_case_reference(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    cases_dir = tmp_path / ".my_coding_agent" / "evals" / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "c1.json").write_text(
        json.dumps(
            {
                "id": "c1",
                "task": "say pong",
                "scorer": "exact_match",
                "expected": {"contains": "pong"},
            }
        )
    )

    path = _write_config(
        tmp_path,
        {
            "run": {"task": "unused for case-only run"},
            "evaluation": {"cases": ["c1"]},
        },
    )
    result, verdict = rcf.execute_from_config(path)

    assert verdict == "pass"
    assert [s.case_id for s in result.scores] == ["c1"]


def test_execute_from_config_with_dataset_reference(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    cases_dir = evals_datasets.DEFAULT_CASES_DIR
    cases_dir.mkdir(parents=True)
    (cases_dir / "d1.json").write_text(
        json.dumps(
            {
                "id": "d1",
                "task": "say pong",
                "scorer": "exact_match",
                "expected": {"contains": "pong"},
            }
        )
    )
    evals_datasets.create_dataset("my-dataset", ["d1"])

    path = _write_config(
        tmp_path,
        {
            "run": {"task": "unused for dataset-only run"},
            "evaluation": {"dataset": "my-dataset"},
        },
    )
    result, verdict = rcf.execute_from_config(path)

    assert verdict == "pass"
    assert [s.case_id for s in result.scores] == ["d1"]

"""Tests for AnomalyDetectNode: unit-level and end-to-end replay through Agent."""

from __future__ import annotations

import json
import unittest.mock as mock

import pytest

from my_coding_agent.engine.llm import LLM
from my_coding_agent.pipeline.context import RunContext
from my_coding_agent.pipeline.nodes.agent import AgentNode
from my_coding_agent.pipeline.nodes.anomaly_detect import (
    STREAK_THRESHOLD,
    AnomalyDetectNode,
    error_signature,
    trailing_streak,
)

# Real bash traceback text, session fbef66a33c18 style: a JSONDecodeError
# raised deep inside urllib, surfaced as stderr in the tool's error text.
_JSON_DECODE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 12, in <module>\n'
    "    data = json.loads(raw)\n"
    '  File ".../json/__init__.py", line 346, in loads\n'
    "    return _default_decoder.decode(s)\n"
    '  File ".../json/decoder.py", line 337, in decode\n'
    "    obj, end = self.raw_decode(s, idx=_w(s, 0).end())\n"
    "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
)

_FILE_NOT_FOUND_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 3, in <module>\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/x.json'"
)


class TestErrorSignature:
    def test_variant_args_same_error_class_same_signature(self) -> None:
        r1 = _failed("bash", _JSON_DECODE_TRACEBACK)
        r2 = _failed(
            "bash",
            _JSON_DECODE_TRACEBACK.replace("line 12", "line 47"),
        )
        assert error_signature(r1) == error_signature(r2)

    def test_different_error_class_different_signature(self) -> None:
        r1 = _failed("bash", _JSON_DECODE_TRACEBACK)
        r2 = _failed("bash", _FILE_NOT_FOUND_TRACEBACK)
        assert error_signature(r1) != error_signature(r2)

    def test_signature_includes_tool_name(self) -> None:
        r1 = _failed("bash", _JSON_DECODE_TRACEBACK)
        r2 = _failed("other_tool", _JSON_DECODE_TRACEBACK)
        assert error_signature(r1) != error_signature(r2)

    def test_extracts_exact_exception_token(self) -> None:
        r = _failed("bash", _JSON_DECODE_TRACEBACK)
        assert error_signature(r) == "bash|json.decoder.JSONDecodeError"

    def test_fallback_bucket_when_no_exception_token(self) -> None:
        r1 = _failed("web_fetch", "connection refused on attempt 1")
        r2 = _failed("web_fetch", "connection refused on attempt 2")
        # digits stripped -> same bucket despite differing attempt numbers
        assert error_signature(r1) == error_signature(r2)

    def test_empty_error_text(self) -> None:
        r = _failed("bash", "")
        assert error_signature(r) == "bash|"


class TestTrailingStreak:
    def test_empty_records_returns_none(self) -> None:
        assert trailing_streak([]) is None

    def test_trailing_success_returns_none(self) -> None:
        records = [_failed("bash", _JSON_DECODE_TRACEBACK), _ok("bash")]
        assert trailing_streak(records) is None

    def test_fail_fail_succeed_stays_below_threshold(self) -> None:
        records = [
            _failed("bash", _JSON_DECODE_TRACEBACK),
            _failed("bash", _JSON_DECODE_TRACEBACK),
            _ok("bash"),
        ]
        assert trailing_streak(records) is None

    def test_alternating_error_classes_never_streak(self) -> None:
        records = [
            _failed("bash", _JSON_DECODE_TRACEBACK),
            _failed("bash", _FILE_NOT_FOUND_TRACEBACK),
            _failed("bash", _JSON_DECODE_TRACEBACK),
        ]
        result = trailing_streak(records)
        assert result is not None
        _, length, _ = result
        assert length == 1

    def test_third_same_signature_failure_crosses_threshold(self) -> None:
        records = [_failed("bash", _JSON_DECODE_TRACEBACK) for _ in range(3)]
        result = trailing_streak(records)
        assert result is not None
        signature, length, indexes = result
        assert length == STREAK_THRESHOLD
        assert signature == "bash|json.decoder.JSONDecodeError"
        assert indexes == [0, 1, 2]

    def test_streak_grows_past_threshold(self) -> None:
        records = [_failed("bash", _JSON_DECODE_TRACEBACK) for _ in range(8)]
        _, length, indexes = trailing_streak(records)
        assert length == 8
        assert indexes == list(range(8))

    @pytest.mark.parametrize("threshold_len", [1, 2])
    def test_below_threshold_lengths_reported_accurately(
        self, threshold_len: int
    ) -> None:
        records = [
            _failed("bash", _JSON_DECODE_TRACEBACK) for _ in range(threshold_len)
        ]
        _, length, _ = trailing_streak(records)
        assert length == threshold_len


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**kwargs) -> RunContext:
    llm = LLM()
    llm.context_window = 1000
    llm._recorder = mock.Mock()
    ctx = object.__new__(RunContext)
    ctx.session_id = "abcdef123456"
    ctx.llm = llm
    ctx.messages = []
    ctx.step_num = 0
    ctx.max_steps = 20
    ctx.last_prompt_tokens = 0
    ctx.context_reset_threshold = 0.75
    ctx.all_tools = []
    ctx.last_response = None
    ctx.signal = "CONTINUE"
    ctx.stop_reason = "max_steps"
    ctx.continuation_messages = []
    ctx.tool_records = []
    ctx.tool_artifacts = {}
    ctx.recorder = mock.Mock()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


_JSON_DECODE_ERROR = "json.decoder.JSONDecodeError: Expecting value: line 1 column 1"


def _failed(name: str = "bash", error: str = _JSON_DECODE_ERROR) -> dict:
    return {"name": name, "ok": False, "error": error}


def _ok(name: str = "bash") -> dict:
    return {"name": name, "ok": True, "error": ""}


def _usage_resp(prompt=10, completion=5, total=15):
    return _Resp(
        {
            "choices": [
                {"message": {"role": "assistant"}, "finish_reason": "tool_calls"}
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            },
        }
    )


# ---------------------------------------------------------------------------
# Unit-level: signaling at threshold, growth, non-interference in isolation
# ---------------------------------------------------------------------------


class TestAnomalyDetectNodeUnit:
    def test_no_signal_below_threshold(self):
        ctx = _make_ctx()
        node = AnomalyDetectNode()
        for _ in range(STREAK_THRESHOLD - 1):
            ctx.tool_records.append(_failed())
            ctx.last_response = _usage_resp()
            node.run(ctx)
        ctx.recorder.record_anomaly.assert_not_called()

    def test_signals_at_third_failure_with_accumulated_tokens(self):
        ctx = _make_ctx()
        node = AnomalyDetectNode()
        for _ in range(3):
            ctx.tool_records.append(_failed())
            ctx.last_response = _usage_resp(prompt=10, completion=5)
            node.run(ctx)
        ctx.recorder.record_anomaly.assert_called_once()
        kwargs = ctx.recorder.record_anomaly.call_args.kwargs
        assert kwargs["streak_len"] == 3
        assert kwargs["tokens_spent"] == 45  # 3 * (10 + 5)
        assert kwargs["tool_name"] == "bash"
        assert kwargs["signature"] == "bash|json.decoder.JSONDecodeError"

    def test_growth_update_reports_grown_length_same_streak_id(self):
        ctx = _make_ctx()
        node = AnomalyDetectNode()
        for _ in range(4):
            ctx.tool_records.append(_failed())
            ctx.last_response = _usage_resp()
            node.run(ctx)
        assert ctx.recorder.record_anomaly.call_count == 2
        first_id = ctx.recorder.record_anomaly.call_args_list[0].kwargs["streak_id"]
        second_call = ctx.recorder.record_anomaly.call_args_list[1].kwargs
        assert second_call["streak_id"] == first_id
        assert second_call["streak_len"] == 4

    def test_fail_fix_retry_never_signals(self):
        ctx = _make_ctx()
        node = AnomalyDetectNode()
        for record in [_failed(), _failed(), _ok()]:
            ctx.tool_records.append(record)
            ctx.last_response = _usage_resp()
            node.run(ctx)
        ctx.recorder.record_anomaly.assert_not_called()

    def test_never_touches_control_signals(self):
        ctx = _make_ctx()
        node = AnomalyDetectNode()
        for _ in range(3):
            ctx.tool_records.append(_failed())
            ctx.last_response = _usage_resp()
            ctx.signal = "CONTINUE"
            ctx.stop_reason = "max_steps"
            messages_before = list(ctx.messages)
            node.run(ctx)
            assert ctx.signal == "CONTINUE"
            assert ctx.stop_reason == "max_steps"
            assert ctx.messages == messages_before

    def test_streak_resets_on_signature_change(self):
        ctx = _make_ctx()
        node = AnomalyDetectNode()
        ctx.tool_records.append(_failed(error=_JSON_DECODE_ERROR))
        ctx.last_response = _usage_resp()
        node.run(ctx)
        ctx.tool_records.append(_failed(error="FileNotFoundError: nope"))
        ctx.last_response = _usage_resp()
        node.run(ctx)
        ctx.tool_records.append(_failed(error=_JSON_DECODE_ERROR))
        ctx.last_response = _usage_resp()
        node.run(ctx)
        # only a length-1 trailing streak at each point -> never reaches threshold
        ctx.recorder.record_anomaly.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end: replay through the real pipeline via Agent
# ---------------------------------------------------------------------------


def _bash_tool_call(call_id: str, command: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def _tool_calls_resp(call_id: str, command: str, prompt=10, completion=5):
    return _Resp(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_bash_tool_call(call_id, command)],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )


def _stop_resp():
    return _Resp(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def _run_streak_agent(monkeypatch, mocker, tmp_path, needs_handback=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    agent = AgentNode(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ],
        tools=[BASH_TOOL_SCHEMA],
        label="TestAgent",
        needs_handback=needs_handback,
    )
    agent.llm.context_window = 4096

    # A failing bash command exits non-zero each time (variant args, same
    # error class), then a clean stop — mirrors the fbef66a33c18 pattern.
    responses = [
        _tool_calls_resp("1", "python3 -c 'import json; json.loads(\"x\")'"),
        _tool_calls_resp("2", "python3 -c 'import json; json.loads(\"y\")'"),
        _tool_calls_resp("3", "python3 -c 'import json; json.loads(\"z\")'"),
        _stop_resp(),
    ]
    mocker.patch.object(agent.llm, "chat_completion", side_effect=responses)
    agent.execute(max_steps=10)
    return agent


def _read_events(agent) -> list[dict]:
    path = agent.recorder.path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestAnomalyDetectNodeEndToEnd:
    def test_replays_streak_and_signals_at_third_failure(
        self, monkeypatch, mocker, tmp_path
    ):
        agent = _run_streak_agent(monkeypatch, mocker, tmp_path)
        events = _read_events(agent)
        anomaly_rows = [e for e in events if e["type"] == "anomaly"]
        assert anomaly_rows, "expected at least one anomaly row"
        final = anomaly_rows[-1]
        assert final["streak_len"] == 3
        assert final["tool_name"] == "bash"
        assert final["tokens_spent"] == 45  # 3 steps * (10 + 5)

    def test_non_interference_identical_messages_signal_stop_reason(
        self, monkeypatch, mocker, tmp_path
    ):
        # Run once with AnomalyDetectNode present (default pipeline).
        agent_with = _run_streak_agent(monkeypatch, mocker, tmp_path / "with")

        # Run again with AnomalyDetectNode removed from the pipeline.
        def _build_without_anomaly(spawn_fn=None, checkpoint_fn=None):
            from my_coding_agent.pipeline.graph import Pipeline
            from my_coding_agent.pipeline.nodes import (
                ContextGuardNode,
                FinalizeStepNode,
                LLMCallNode,
                ToolDispatchNode,
            )

            return Pipeline(
                [
                    ContextGuardNode(spawn_fn=spawn_fn),
                    LLMCallNode(),
                    ToolDispatchNode(),
                    FinalizeStepNode(),
                ],
                checkpoint_fn=checkpoint_fn,
            )

        mocker.patch(
            "my_coding_agent.pipeline.build_default_pipeline",
            side_effect=_build_without_anomaly,
        )
        agent_without = _run_streak_agent(monkeypatch, mocker, tmp_path / "without")

        assert agent_with.messages == agent_without.messages
        assert agent_with.stop_reason == agent_without.stop_reason
        chat_with = agent_with.llm.chat_completion
        chat_without = agent_without.llm.chat_completion
        assert chat_with.call_count == chat_without.call_count == 4

    def test_subagent_writes_own_anomaly_rows(self, monkeypatch, mocker, tmp_path):
        agent = _run_streak_agent(monkeypatch, mocker, tmp_path, needs_handback=True)
        events_path = tmp_path / ".my_coding_agent" / agent.session_id / "events.jsonl"
        assert events_path.exists()
        rows = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert any(r.get("type") == "anomaly" for r in rows)

    def test_clean_session_zero_false_flags(self, monkeypatch, mocker, tmp_path):
        monkeypatch.chdir(tmp_path)
        agent = AgentNode(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
            ],
            tools=[BASH_TOOL_SCHEMA],
            label="CleanAgent",
        )
        agent.llm.context_window = 4096
        responses = [
            _tool_calls_resp("1", "echo one"),
            _tool_calls_resp("2", "echo two"),
            _stop_resp(),
        ]
        mocker.patch.object(agent.llm, "chat_completion", side_effect=responses)
        agent.execute(max_steps=10)
        events = _read_events(agent)
        assert not [e for e in events if e["type"] == "anomaly"]

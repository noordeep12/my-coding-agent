"""Regression tests for resources capture on timed events (node-resource-monitoring).

Verifies rows omit `resources` when capture is unavailable, and carry it
(with the machine-wide marker) when the sampler has data.
"""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import Recorder

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_RESPONSE = {"choices": [{"message": {"content": "hi"}}]}


def _make_recorder(tmp_path):
    sid = "res00001"
    sdir = tmp_path / sid
    sdir.mkdir()
    return Recorder(session_id=sid, session_dir=sdir), sdir / "events.jsonl"


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestLlmCallResources:
    def test_resources_key_omitted_when_none(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_llm_call(
            kind="main",
            call=1,
            latency_s=0.01,
            usage=_USAGE,
            messages=[{"role": "user", "content": "hi"}],
            context_window=8192,
            response_data=_RESPONSE,
            resources=None,
        )
        ev = _read_events(path)[-1]
        assert "resources" not in ev

    def test_resources_written_when_present(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        resources = {
            "machine_wide": True,
            "sample_count": 3,
            "ram_pct": {"avg": 1.0, "peak": 2.0},
        }
        rec.record_llm_call(
            kind="main",
            call=1,
            latency_s=0.01,
            usage=_USAGE,
            messages=[{"role": "user", "content": "hi"}],
            context_window=8192,
            response_data=_RESPONSE,
            resources=resources,
        )
        ev = _read_events(path)[-1]
        assert ev["resources"] == resources


class TestSummarizerResources:
    def test_resources_key_omitted_when_none(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_summarizer(
            kind="handoff",
            step=1,
            triggered_by="context_guard",
            latency_s=0.02,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            resources=None,
        )
        ev = _read_events(path)[-1]
        assert "resources" not in ev


class TestToolCallResources:
    def test_resources_computed_from_recorder_sampler(self, tmp_path, monkeypatch):
        rec, path = _make_recorder(tmp_path)
        fixed = {
            "machine_wide": True,
            "sample_count": 1,
            "ram_pct": {"avg": 5.0, "peak": 5.0},
        }
        monkeypatch.setattr(rec._sampler, "summarize_window", lambda a, b: fixed)
        rec.before_tool("bash", {"command": "echo hi"})
        rec.after_tool("bash", {"command": "echo hi"}, "hi", True, None)
        ev = _read_events(path)[-1]
        assert ev["resources"] == fixed

    def test_resources_absent_when_sampler_disabled(self, tmp_path, monkeypatch):
        rec, path = _make_recorder(tmp_path)
        monkeypatch.setattr(rec._sampler, "summarize_window", lambda a, b: None)
        rec.before_tool("bash", {"command": "echo hi"})
        rec.after_tool("bash", {"command": "echo hi"}, "hi", True, None)
        ev = _read_events(path)[-1]
        assert "resources" not in ev


class TestResourceRollup:
    def test_resource_rollup_delegates_to_sampler(self, tmp_path, monkeypatch):
        rec, _ = _make_recorder(tmp_path)
        fixed = {"machine_wide": True, "sample_count": 4}
        monkeypatch.setattr(rec._sampler, "session_rollup", lambda: fixed)
        assert rec.resource_rollup() == fixed

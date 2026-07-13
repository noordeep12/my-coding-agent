"""Lifecycle hook firing at the tool-dispatch choke point (issue #129).

Mirrors ``test_tool_refusal_gate.py``'s structure: a blocking ``PreToolUse``
hook never lets a subprocess spawn, on both dispatch paths and in a subagent;
a non-blocking hook leaves the envelope byte-identical; and a hook-free run
behaves exactly as it did before this seam existed (zero-config parity).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from my_coding_agent.engine.hooks import Hooks
from my_coding_agent.engine.hooks.schema import (
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
    HookResult,
    HookSpec,
)
from my_coding_agent.engine.tool_execution import ToolExecutor
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry


def _tool_call(name, args, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _blocking_spec(name="danger-gate", tool="bash", reason="blocked for test"):
    return HookSpec(
        name=name,
        event=EVENT_PRE_TOOL_USE,
        tool=tool,
        callback=lambda ctx: HookResult.block(reason),
    )


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    """Fail loudly if a blocked command ever reaches ``subprocess.run``."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            f"subprocess.run must never be called for a blocked command: {args!r}"
        )

    monkeypatch.setattr(subprocess, "run", _boom)


class _RecordingRecorder:
    """Captures ``record_hook`` calls; ``after_tool``/``record_refusal`` are no-ops."""

    def __init__(self):
        self.hook_calls: list[dict] = []

    def record_hook(self, **kw):
        self.hook_calls.append(kw)

    def after_tool(self, *a, **kw):
        return None

    def record_refusal(self, **kw):
        return None


class TestPreToolUseBlocksExecution:
    def test_blocked_command_spawns_no_subprocess(self, bare_executor):
        bare_executor.hooks = Hooks([_blocking_spec()])
        raw, failure = bare_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        assert raw is None
        assert failure is not None
        assert failure["reason"] == "blocked_by_hook"
        assert failure["hook_name"] == "danger-gate"
        assert failure["block_reason"] == "blocked for test"

    def test_non_matching_tool_is_unaffected(self, bare_executor, monkeypatch):
        bare_executor.hooks = Hooks([_blocking_spec(tool="write_file")])
        monkeypatch.setattr(ToolsRegistry, "bash", lambda self, **kw: "hello")
        raw, failure = bare_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        assert failure is None
        assert raw == "hello"

    def test_sequential_dispatch_blocks_and_appends_error_result(self, bare_executor):
        bare_executor.hooks = Hooks([_blocking_spec()])
        bare_executor.tool_calls = [_tool_call("bash", {"command": "echo hi"}, "c0")]
        messages, records = bare_executor.run()
        assert messages[0]["status"] == "error"
        env = json.loads(messages[0]["content"])
        assert env["ok"] is False
        assert env["metadata"]["reason"] == "blocked_by_hook"
        assert env["metadata"]["hook_block"]["hook_name"] == "danger-gate"
        assert records[0]["ok"] is False

    def test_concurrent_worker_path_shares_the_same_gate(self, bare_executor):
        bare_executor.hooks = Hooks([_blocking_spec()])
        raw, failure, *_ = bare_executor._invoke_timed(
            "c0", "bash", {"command": "echo hi"}
        )
        assert raw is None
        assert failure["reason"] == "blocked_by_hook"

    def test_subagent_executor_inherits_the_same_gate(self, bare_llm, silent_logger):
        sub_executor = ToolExecutor({"tool_calls": []}, bare_llm)
        sub_executor.logger = silent_logger
        sub_executor.hooks = Hooks([_blocking_spec()])
        raw, failure = sub_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        assert raw is None
        assert failure["reason"] == "blocked_by_hook"


class TestBlockEnvelope:
    def test_envelope_is_ok_false_with_structured_metadata(self, bare_executor):
        bare_executor.hooks = Hooks([_blocking_spec()])
        _, failure = bare_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        content, status, record = bare_executor.after_tool_call(
            "c0", "bash", {"command": "echo hi"}, None, failure
        )
        env = json.loads(content)
        assert env["ok"] is False
        assert status == "error"
        assert record["ok"] is False
        assert "Blocked by hook" in env["error"]
        assert "danger-gate" in env["error"]
        assert env["metadata"]["hook_block"]["reason"] == "blocked for test"

    def test_non_blocking_hook_envelope_is_byte_identical(
        self, bare_executor, monkeypatch
    ):
        observed = []
        observe_spec = HookSpec(
            name="observer",
            event=EVENT_PRE_TOOL_USE,
            callback=lambda ctx: observed.append(ctx) or None,
        )
        monkeypatch.setattr(ToolsRegistry, "bash", lambda self, **kw: "hello")

        bare_executor.hooks = Hooks([])
        raw, failure = bare_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        baseline, status_baseline, _ = bare_executor.after_tool_call(
            "c0", "bash", {"command": "echo hi"}, raw, failure
        )

        bare_executor.hooks = Hooks([observe_spec])
        raw2, failure2 = bare_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        with_hook, status_with_hook, _ = bare_executor.after_tool_call(
            "c0", "bash", {"command": "echo hi"}, raw2, failure2
        )

        assert with_hook == baseline
        assert status_with_hook == status_baseline
        assert len(observed) == 1


class TestHookEventChannel:
    def test_blocking_pretooluse_hook_is_recorded(self, bare_executor):
        recorder = _RecordingRecorder()
        bare_executor.llm._recorder = recorder
        bare_executor.hooks = Hooks([_blocking_spec()])
        bare_executor.step_num = 3
        hook_firings = []
        raw, failure = bare_executor.invoke_tool(
            "c0", "bash", {"command": "echo hi"}, hook_firings
        )
        bare_executor.after_tool_call(
            "c0",
            "bash",
            {"command": "echo hi"},
            raw,
            failure,
            hook_firings=hook_firings,
        )
        pre_rows = [c for c in recorder.hook_calls if c["event"] == EVENT_PRE_TOOL_USE]
        assert len(pre_rows) == 1
        assert pre_rows[0]["outcome"] == "blocked"
        assert pre_rows[0]["reason"] == "blocked for test"
        assert pre_rows[0]["hook_name"] == "danger-gate"
        assert pre_rows[0]["step"] == 3
        post_rows = [
            c for c in recorder.hook_calls if c["event"] == EVENT_POST_TOOL_USE
        ]
        assert post_rows == []

    def test_allowing_hook_is_recorded_fired_and_post_hook_fires(
        self, bare_executor, monkeypatch
    ):
        recorder = _RecordingRecorder()
        bare_executor.llm._recorder = recorder
        monkeypatch.setattr(ToolsRegistry, "bash", lambda self, **kw: "hello")
        pre_spec = HookSpec(
            name="observer-pre", event=EVENT_PRE_TOOL_USE, callback=lambda ctx: None
        )
        post_spec = HookSpec(
            name="observer-post", event=EVENT_POST_TOOL_USE, callback=lambda ctx: None
        )
        bare_executor.hooks = Hooks([pre_spec, post_spec])
        hook_firings = []
        raw, failure = bare_executor.invoke_tool(
            "c0", "bash", {"command": "echo hi"}, hook_firings
        )
        bare_executor.after_tool_call(
            "c0",
            "bash",
            {"command": "echo hi"},
            raw,
            failure,
            hook_firings=hook_firings,
        )
        assert len(recorder.hook_calls) == 2
        outcomes = {c["event"]: c["outcome"] for c in recorder.hook_calls}
        assert outcomes[EVENT_PRE_TOOL_USE] == "fired"
        assert outcomes[EVENT_POST_TOOL_USE] == "fired"

    def test_hook_free_run_emits_no_hook_rows(self, bare_executor, monkeypatch):
        recorder = _RecordingRecorder()
        bare_executor.llm._recorder = recorder
        monkeypatch.setattr(ToolsRegistry, "bash", lambda self, **kw: "hello")
        bare_executor.hooks = Hooks([])
        hook_firings = []
        raw, failure = bare_executor.invoke_tool(
            "c0", "bash", {"command": "echo hi"}, hook_firings
        )
        bare_executor.after_tool_call(
            "c0",
            "bash",
            {"command": "echo hi"},
            raw,
            failure,
            hook_firings=hook_firings,
        )
        assert recorder.hook_calls == []


class TestZeroConfigParity:
    def test_no_env_var_yields_no_hooks_and_no_events(self, monkeypatch):
        from my_coding_agent.engine.hooks import HOOKS_CONFIG_ENV_VAR

        monkeypatch.delenv(HOOKS_CONFIG_ENV_VAR, raising=False)
        # A fresh registry with no config set loads zero hooks.
        hooks = Hooks.load()
        assert hooks._specs == []

    def test_executor_with_no_hooks_configured_matches_pre_change_envelope(
        self, bare_executor, monkeypatch
    ):
        """No ``PreToolUse``/``PostToolUse`` hooks configured → the envelope
        and dispatch behavior are identical to a run before this seam existed
        (the #124 refusal-gate characterization tests already lock this
        shape; this only re-asserts it holds with ``self.hooks`` present but
        empty)."""
        monkeypatch.setattr(ToolsRegistry, "bash", lambda self, **kw: "hello")
        assert bare_executor.hooks._specs == []
        raw, failure = bare_executor.invoke_tool("c0", "bash", {"command": "echo hi"})
        content, status, _ = bare_executor.after_tool_call(
            "c0", "bash", {"command": "echo hi"}, raw, failure
        )
        env = json.loads(content)
        assert env["ok"] is True
        assert status == "success"
        assert "hook_block" not in env["metadata"]


class TestReusabilityDemonstration:
    def test_a_blocking_policy_expressed_as_a_hook_enforces_without_engine_edits(
        self, bare_executor
    ):
        """A security policy (block a sentinel command) is expressed purely as
        a registered hook — no change to ``invoke_tool``'s own logic."""

        def deny_sentinel(ctx):
            if ctx.args and ctx.args.get("command") == "sentinel-danger-command":
                return HookResult.block("sentinel command is not allowed")
            return None

        bare_executor.hooks = Hooks(
            [
                HookSpec(
                    name="sentinel-policy",
                    event=EVENT_PRE_TOOL_USE,
                    tool="bash",
                    callback=deny_sentinel,
                )
            ]
        )
        raw, failure = bare_executor.invoke_tool(
            "c0", "bash", {"command": "sentinel-danger-command"}
        )
        assert raw is None
        assert failure["reason"] == "blocked_by_hook"
        assert failure["block_reason"] == "sentinel command is not allowed"


class TestSessionHooksFireOnce:
    def test_session_start_and_end_fire_exactly_once(self):
        from my_coding_agent.pipeline.nodes.agent import AgentNode

        calls = {"start": 0, "end": 0}

        def on_start(ctx):
            calls["start"] += 1
            return None

        def on_end(ctx):
            calls["end"] += 1
            return None

        agent = object.__new__(AgentNode)
        agent.hooks = Hooks(
            [
                HookSpec(
                    name="start-counter", event=EVENT_SESSION_START, callback=on_start
                ),
                HookSpec(name="end-counter", event=EVENT_SESSION_END, callback=on_end),
            ]
        )
        agent.session_id = "sess-test"
        agent.step_num = 0
        recorder = _RecordingRecorder()
        agent.recorder = recorder

        agent._fire_session_hook(EVENT_SESSION_START)
        agent._fire_session_hook(EVENT_SESSION_END)

        assert calls == {"start": 1, "end": 1}
        events = [c["event"] for c in recorder.hook_calls]
        assert events == [EVENT_SESSION_START, EVENT_SESSION_END]

"""Tests for the lifecycle hook registry (issue #129)."""

import json

from my_coding_agent.engine.hooks import HOOKS_CONFIG_ENV_VAR, Hooks
from my_coding_agent.engine.hooks.schema import (
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    HookContext,
    HookResult,
    HookSpec,
)


def _ctx(
    event: str = EVENT_PRE_TOOL_USE, tool_name: str | None = "bash"
) -> HookContext:
    return HookContext(event=event, session_id="sess1", step=1, tool_name=tool_name)


def test_fire_matches_event_and_tool():
    calls = []
    spec = HookSpec(
        name="observer",
        event=EVENT_PRE_TOOL_USE,
        tool="bash",
        callback=lambda ctx: calls.append(ctx) or None,
    )
    hooks = Hooks([spec])

    firings = hooks.fire(EVENT_PRE_TOOL_USE, _ctx(tool_name="bash"))
    assert len(firings) == 1
    assert calls == [_ctx(tool_name="bash")]

    calls.clear()
    firings = hooks.fire(EVENT_PRE_TOOL_USE, _ctx(tool_name="write_file"))
    assert firings == []
    assert calls == []


def test_fire_ignores_other_events():
    spec = HookSpec(
        name="post-only", event=EVENT_POST_TOOL_USE, callback=lambda ctx: None
    )
    hooks = Hooks([spec])
    assert hooks.fire(EVENT_PRE_TOOL_USE, _ctx()) == []


def test_fire_runs_in_registration_order():
    order = []
    spec_a = HookSpec(
        name="a", event=EVENT_PRE_TOOL_USE, callback=lambda ctx: order.append("a")
    )
    spec_b = HookSpec(
        name="b", event=EVENT_PRE_TOOL_USE, callback=lambda ctx: order.append("b")
    )
    hooks = Hooks([spec_a, spec_b])
    hooks.fire(EVENT_PRE_TOOL_USE, _ctx())
    assert order == ["a", "b"]


def test_block_short_circuits_remaining_pre_tool_use_hooks():
    order = []

    def blocker(ctx):
        order.append("blocker")
        return HookResult.block("nope")

    def never_runs(ctx):
        order.append("never")
        return None

    hooks = Hooks(
        [
            HookSpec(name="blocker", event=EVENT_PRE_TOOL_USE, callback=blocker),
            HookSpec(name="never", event=EVENT_PRE_TOOL_USE, callback=never_runs),
        ]
    )
    firings = hooks.fire(EVENT_PRE_TOOL_USE, _ctx())
    assert order == ["blocker"]
    assert len(firings) == 1
    assert firings[0][1].blocked is True
    assert firings[0][1].reason == "nope"


def test_raising_hook_is_fail_open():
    def boom(ctx):
        raise RuntimeError("boom")

    hooks = Hooks([HookSpec(name="boom", event=EVENT_PRE_TOOL_USE, callback=boom)])
    firings = hooks.fire(EVENT_PRE_TOOL_USE, _ctx())
    assert len(firings) == 1
    assert firings[0][1].blocked is False


def test_load_with_no_env_var_yields_empty_registry(monkeypatch):
    monkeypatch.delenv(HOOKS_CONFIG_ENV_VAR, raising=False)
    hooks = Hooks.load()
    assert hooks.fire(EVENT_PRE_TOOL_USE, _ctx()) == []


def test_load_with_missing_file_yields_empty_registry(monkeypatch, tmp_path):
    monkeypatch.setenv(HOOKS_CONFIG_ENV_VAR, str(tmp_path / "missing.json"))
    hooks = Hooks.load()
    assert hooks.fire(EVENT_PRE_TOOL_USE, _ctx()) == []


def test_load_parses_valid_entry(monkeypatch, tmp_path):
    config_path = tmp_path / "hooks.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "sample",
                    "event": "PreToolUse",
                    "tool": "bash",
                    "callable": "my_coding_agent.engine.hooks.schema:HookResult",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(HOOKS_CONFIG_ENV_VAR, str(config_path))
    hooks = Hooks.load()
    assert len(hooks._specs) == 1
    assert hooks._specs[0].name == "sample"


def test_load_skips_malformed_entries(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "hooks.json"
    config_path.write_text(
        json.dumps(
            [
                {"name": "bad-event", "event": "NoSuchEvent", "callable": "os:getcwd"},
                {"event": "PreToolUse", "callable": "os:getcwd"},  # missing name
                {
                    "name": "bad-callable",
                    "event": "PreToolUse",
                    "callable": "not-a-target",
                },
                {
                    "name": "bad-import",
                    "event": "PreToolUse",
                    "callable": "nope.nope:nope",
                },
                "not-a-dict",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(HOOKS_CONFIG_ENV_VAR, str(config_path))
    hooks = Hooks.load()
    assert hooks._specs == []


def test_load_rejects_non_array_config(monkeypatch, tmp_path):
    config_path = tmp_path / "hooks.json"
    config_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    monkeypatch.setenv(HOOKS_CONFIG_ENV_VAR, str(config_path))
    hooks = Hooks.load()
    assert hooks._specs == []

"""Tests for task-scoped system prompt composition (main agent + subagent)."""

import re

from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.cli import _all_tools, _system_prompt

_TIMESTAMP_RE = re.compile(r"^Current date and time: .+$")


def _delegate_system_prompt(mocker):
    """Run delegate() with a mocked AgentNode and return the system prompt used."""
    fake_agent = mocker.Mock()
    fake_agent.session_id = "abc123"
    fake_agent.stop_reason = "stop"
    fake_agent.final_assistant_text.return_value = "report"
    fake_agent.handback_report = None
    captured = {}

    def _fake_agent_node(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return fake_agent

    mocker.patch("my_coding_agent.engine.agent.AgentNode", side_effect=_fake_agent_node)
    ToolsRegistry().delegate(task="do X", known_facts="ctx")
    return captured["messages"][0]["content"]


def test_main_prompt_byte_stable_before_timestamp():
    p1 = _system_prompt()
    p2 = _system_prompt()
    prefix1, _, _ = p1.rpartition("\n")
    prefix2, _, _ = p2.rpartition("\n")
    assert prefix1 == prefix2


def test_main_prompt_has_no_prose_tool_docs():
    prompt = _system_prompt()
    for t in _all_tools():
        name = t["function"]["name"]
        assert f"{name}(" not in prompt


def test_main_prompt_has_no_prose_tool_docs_regardless_of_registry():
    prompt = _system_prompt()
    assert "Available tools" not in prompt


def test_main_prompt_stable_core_content():
    prompt = _system_prompt()
    import os

    assert os.getcwd() in prompt
    assert "schema_version" in prompt
    import platform

    assert platform.system() in prompt


def test_main_prompt_has_no_volatile_repo_state():
    prompt = _system_prompt()
    assert "git status" not in prompt.lower()
    assert "workspace:" not in prompt.lower()
    assert "commit history" not in prompt.lower()
    assert "contents :" not in prompt.lower()


def test_main_prompt_timestamp_is_final_line():
    prompt = _system_prompt()
    last_line = prompt.rstrip("\n").splitlines()[-1]
    assert _TIMESTAMP_RE.match(last_line)


def test_subagent_prompt_has_working_directory_and_envelope(mocker):
    prompt = _delegate_system_prompt(mocker)
    import os

    assert os.getcwd() in prompt
    assert "schema_version" in prompt


def test_subagent_prompt_timestamp_is_final_line(mocker):
    prompt = _delegate_system_prompt(mocker)
    last_line = prompt.rstrip("\n").splitlines()[-1]
    assert _TIMESTAMP_RE.match(last_line)


def test_subagent_prompt_states_handback_is_verbatim(mocker):
    prompt = _delegate_system_prompt(mocker)
    assert "returned verbatim to the delegating agent" in prompt
    assert "report expectations" in prompt


def test_subagent_prompt_byte_stable_before_timestamp(mocker):
    p1 = _delegate_system_prompt(mocker)
    p2 = _delegate_system_prompt(mocker)
    prefix1, _, _ = p1.rpartition("\n")
    prefix2, _, _ = p2.rpartition("\n")
    assert prefix1 == prefix2


def test_roles_have_distinct_identities(mocker):
    main_prompt = _system_prompt()
    sub_prompt = _delegate_system_prompt(mocker)
    assert "coding assistant working in a terminal" in main_prompt
    assert "focused subagent working for a main coding assistant" in sub_prompt
    assert main_prompt != sub_prompt

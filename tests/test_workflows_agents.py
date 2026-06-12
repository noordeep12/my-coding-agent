"""Behavioral tests for the workflows/ CLI helpers and agents/ helpers.

These packages ship in the wheel but had no test exercise. The tests below
target the pure/near-pure helpers (git wrappers, tool listing, prompt building,
session selection) and the ``delegate`` tool dispatch, mocking subprocess and
the network-touching ``Agent`` so nothing hits a real LLM or runs real git.
"""

import subprocess

import pytest
from agents import discovery, session_analyzer
from workflows import main as wf_main

from my_coding_agent.tools import ToolsRegistry

# --- _git wrappers (identical helper across the three modules) ----------------


@pytest.mark.parametrize("module", [wf_main, discovery, session_analyzer])
def test_git_returns_stripped_stdout(module, mocker):
    mocker.patch(
        f"{module.__name__}.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="main\n", stderr=""),
    )
    assert module._git("rev-parse", "--abbrev-ref", "HEAD") == "main"


@pytest.mark.parametrize("module", [wf_main, discovery, session_analyzer])
def test_git_returns_empty_on_oserror(module, mocker):
    mocker.patch(f"{module.__name__}.subprocess.run", side_effect=FileNotFoundError("no git"))
    assert module._git("status") == ""


# --- workflows._all_tools ----------------------------------------------------


def test_all_tools_returns_public_tool_defs():
    tools = wf_main._all_tools()
    names = {t["function"]["name"] for t in tools}
    # Public ToolsRegistry methods become tools; private (underscore) ones do not.
    assert {"bash", "read_file", "write_file"} <= names
    assert not any(n.startswith("_") for n in names)


# --- workflows._system_prompt ------------------------------------------------


def test_system_prompt_includes_workspace_and_tools(mocker):
    mocker.patch("workflows.main._git", return_value="clean")
    mocker.patch("workflows.main.os.getcwd", return_value="/fake/ws")
    mocker.patch("workflows.main.os.listdir", return_value=["a.py"])
    mocker.patch("workflows.main.Path.exists", return_value=False)

    tools = [
        {"function": {"name": "bash", "description": "run a command",
                      "parameters": {"properties": {"command": {}}}}},
    ]
    prompt = wf_main._system_prompt(tools)
    assert "/fake/ws" in prompt
    assert "bash(command)" in prompt
    assert "run with --discover" in prompt  # discovery.md absent branch


# --- agents.session_analyzer._most_recent_session ----------------------------


def test_most_recent_session_none_when_no_sessions(tmp_path, mocker):
    mocker.patch.object(session_analyzer, "_BASE_DIR", tmp_path)
    assert session_analyzer._most_recent_session() is None


def test_most_recent_session_picks_newest(tmp_path, mocker):
    mocker.patch.object(session_analyzer, "_BASE_DIR", tmp_path)
    old = tmp_path / "old_session"
    new = tmp_path / "new_session"
    for d in (old, new):
        d.mkdir()
        (d / "session_data.json").write_text("{}")
    # Make `new` strictly more recent.
    import os
    os.utime(old / "session_data.json", (1000, 1000))
    os.utime(new / "session_data.json", (2000, 2000))
    assert session_analyzer._most_recent_session() == "new_session"


# --- ToolsRegistry.delegate (spawns a subagent) ------------------------------


def test_delegate_returns_subagent_report(mocker):
    fake_agent = mocker.Mock()
    fake_agent.run.return_value = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "the report"},
    ]
    mocker.patch("my_coding_agent.agent.Agent", return_value=fake_agent)

    reg = ToolsRegistry(tools=[{"function": {"name": "delegate"}}, {"function": {"name": "bash"}}])
    out = reg.delegate(task="explore x", context="paths here")

    assert out == "the report"
    fake_agent.run.assert_called_once_with(max_steps=5)


def test_delegate_handles_no_report(mocker):
    fake_agent = mocker.Mock()
    fake_agent.run.return_value = [{"role": "user", "content": "task"}]  # no assistant content
    mocker.patch("my_coding_agent.agent.Agent", return_value=fake_agent)

    reg = ToolsRegistry(tools=[{"function": {"name": "delegate"}}])
    assert reg.delegate(task="t", context="c") == "(subagent produced no report)"

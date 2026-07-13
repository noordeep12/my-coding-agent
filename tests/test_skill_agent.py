"""Skill placement in AgentNode: opening-message index, continuation, delegate."""

import json
from pathlib import Path

from my_coding_agent.engine.checkpoint import Checkpoint
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.engine.tool_registry.skills import Skill
from my_coding_agent.pipeline.nodes.agent import AgentNode
from my_coding_agent.utils import detach_session_log

_SYSTEM = {"role": "system", "content": "SYSTEM PROMPT (stable)"}
_USER = {"role": "user", "content": "Do the task."}


def _make_node(monkeypatch, tmp_path, **kwargs) -> AgentNode:
    """Construct a real, network-free AgentNode inside an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    agent = AgentNode(
        messages=[dict(_SYSTEM), dict(_USER)],
        tools=[],
        label="Test",
        **kwargs,
    )
    return agent


def _cleanup(agent: AgentNode) -> None:
    detach_session_log(agent._session_log_handler)


# ── placement: no skills → byte-identical opening message ─────────────────────


def test_no_skills_opening_message_byte_identical(monkeypatch, tmp_path):
    agent = _make_node(monkeypatch, tmp_path, skills={})
    try:
        assert agent.messages[0]["content"] == _SYSTEM["content"]
        assert agent.messages[1]["content"] == _USER["content"]
        assert agent._rendered_index is None
    finally:
        _cleanup(agent)


# ── placement: skills → index appended after the task text ────────────────────


def test_skills_index_appended_to_opening_message(monkeypatch, tmp_path):
    skills = {"a": Skill("a", "does a", "body a"), "b": Skill("b", "does b", "body b")}
    agent = _make_node(monkeypatch, tmp_path, skills=skills)
    try:
        # System prompt untouched (#75 invariant): index never in the system msg.
        assert agent.messages[0]["content"] == _SYSTEM["content"]
        user = agent.messages[1]["content"]
        assert user.startswith("Do the task.")  # original task text preserved first
        assert "use_skill" in user
        assert "- a: does a" in user
        assert "- b: does b" in user
        assert agent._rendered_index is not None
        assert agent._rendered_index.names == ["a", "b"]
    finally:
        _cleanup(agent)


def test_no_skills_body_not_present(monkeypatch, tmp_path):
    # With no skills, no skill body / index text appears anywhere.
    agent = _make_node(monkeypatch, tmp_path, skills={})
    try:
        joined = " ".join(m["content"] for m in agent.messages)
        assert "use_skill" not in joined
        assert "Available skills" not in joined
    finally:
        _cleanup(agent)


# ── continuation seeding: loaded bodies re-injected, dedup seeded ──────────────


def test_continuation_reinjects_loaded_bodies(monkeypatch, tmp_path):
    skills = {
        "a": Skill("a", "does a", "FULL BODY A"),
        "b": Skill("b", "does b", "FULL BODY B"),
    }
    agent = _make_node(monkeypatch, tmp_path, skills=skills, loaded_skills={"a"})
    try:
        user = agent.messages[1]["content"]
        assert "FULL BODY A" in user  # loaded skill re-injected (D6)
        assert "FULL BODY B" not in user  # unloaded stays index-only
        assert "- b: does b" in user  # but b still listed in the index
        # The seeded loaded-set is preserved so post-reset dedup keeps working.
        assert agent.loaded_skills == {"a"}
    finally:
        _cleanup(agent)


# ── delegate parity (D7) ──────────────────────────────────────────────────────


def _capture_delegate_child(mocker, skills):
    """Run delegate() with a mocked child AgentNode; return the captured kwargs."""
    fake = mocker.Mock()
    fake.session_id = "child0001"
    fake.stop_reason = "stop"
    fake.final_assistant_text.return_value = "report text"
    fake.handback_report = None
    captured = {}

    def _fake_node(*args, **kwargs):
        captured.update(kwargs)
        return fake

    mocker.patch(
        "my_coding_agent.pipeline.nodes.agent.AgentNode", side_effect=_fake_node
    )
    reg = ToolsRegistry(skills=skills)
    reg._tools = [{"type": "function", "function": {"name": "use_skill"}}]
    reg.delegate(task="do X")
    return captured


def test_delegate_passes_snapshot_to_child(mocker):
    skills = {"a": Skill("a", "does a", "body")}
    captured = _capture_delegate_child(mocker, skills)
    assert captured["skills"] is skills  # same snapshot, no disk re-scan (D7)


def test_delegate_child_has_own_empty_loaded_set(mocker):
    # delegate must NOT pass loaded_skills — the child starts fresh (D7).
    captured = _capture_delegate_child(mocker, {"a": Skill("a", "d", "b")})
    assert "loaded_skills" not in captured


def test_skill_free_delegate_child_byte_identical(mocker):
    captured = _capture_delegate_child(mocker, {})
    # Empty snapshot passed through → the child places no index; its opening
    # message is the task only, byte-identical to a pre-skills delegate.
    assert captured["skills"] == {}
    assert captured["messages"][1]["content"] == "do X"


# ── resume seeding (D5): skills service use_skill without re-placing index ─────


def test_resumed_agent_has_skills_but_does_not_replace_index(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    skills = {"a": Skill("a", "does a", "body a")}
    # The dead session's opening message already carried the index; the resumed
    # conversation must not gain a second copy of it.
    checkpoint = Checkpoint(
        session_id="deadbeef1234",
        step_num=4,
        last_prompt_tokens=42,
        messages=[dict(_SYSTEM), {"role": "user", "content": "Do the task."}],
    )
    agent = AgentNode.from_checkpoint(checkpoint, tools=[], skills=skills)
    try:
        # Registry can service use_skill on a resumed run.
        assert agent.skills == skills
        # Index NOT re-placed: opening message unchanged, no offered event queued.
        assert agent.messages[1]["content"] == "Do the task."
        assert agent._rendered_index is None
    finally:
        _cleanup(agent)


def test_resumed_agent_emits_no_second_skill_index_event(monkeypatch, tmp_path, mocker):
    monkeypatch.chdir(tmp_path)
    skills = {"a": Skill("a", "does a", "body a")}
    checkpoint = Checkpoint(
        session_id="deadbeef1234",
        step_num=1,
        last_prompt_tokens=0,
        messages=[dict(_SYSTEM), {"role": "user", "content": "Do the task."}],
    )
    agent = AgentNode.from_checkpoint(checkpoint, tools=[], skills=skills)
    _stub_execute(agent, mocker)
    agent.execute(max_steps=2)
    events_path = Path(".my_coding_agent") / agent.session_id / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert not any(e["type"] == "skill_index" for e in events)


# ── session-start event via real execute ──────────────────────────────────────


def _stub_execute(agent, mocker):
    mocker.patch("my_coding_agent.pipeline.nodes.agent.print_banner")
    mocker.patch("my_coding_agent.pipeline.nodes.agent.print_run_summary")
    agent.llm.context_window = 1000
    resp = type(
        "R",
        (),
        {
            "json": lambda self: {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        },
    )()
    mocker.patch.object(agent.llm, "chat_completion", return_value=resp)


def test_execute_emits_skill_index_event_at_session_start(
    monkeypatch, tmp_path, mocker
):
    skills = {"a": Skill("a", "does a", "b"), "b": Skill("b", "does b", "c")}
    agent = _make_node(monkeypatch, tmp_path, skills=skills)
    _stub_execute(agent, mocker)
    agent.execute(max_steps=1)
    events_path = Path(".my_coding_agent") / agent.session_id / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    skill_events = [e for e in events if e["type"] == "skill_index"]
    assert len(skill_events) == 1
    assert skill_events[0]["names"] == ["a", "b"]


def test_execute_no_skills_emits_no_skill_event(monkeypatch, tmp_path, mocker):
    agent = _make_node(monkeypatch, tmp_path, skills={})
    _stub_execute(agent, mocker)
    agent.execute(max_steps=1)
    events_path = Path(".my_coding_agent") / agent.session_id / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert not any(e["type"] == "skill_index" for e in events)

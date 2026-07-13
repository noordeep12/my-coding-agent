"""Tests for the ``use_skill`` tool: lazy load, dedup, error, offload."""

from my_coding_agent.engine.tool_execution.concurrency import is_parallel_safe
from my_coding_agent.engine.tool_registry import ARTIFACT_THRESHOLD
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.engine.tool_registry.skills import Skill


def _reg(skills, loaded=None):
    return ToolsRegistry(
        skills=skills, loaded_skills=loaded if loaded is not None else set()
    )


# ── lazy load / dedup / error ─────────────────────────────────────────────────


def test_use_skill_loads_named_body_only():
    skills = {
        "a": Skill("a", "does a", "BODY OF A"),
        "b": Skill("b", "does b", "BODY OF B"),
    }
    reg = _reg(skills)
    out = reg.use_skill("a")
    assert isinstance(out, str)
    assert out.startswith("Skill: a")
    assert "BODY OF A" in out
    assert "BODY OF B" not in out


def test_use_skill_marks_loaded_set():
    loaded: set[str] = set()
    reg = _reg({"a": Skill("a", "d", "body")}, loaded)
    reg.use_skill("a")
    assert loaded == {"a"}


def test_use_skill_repeat_returns_already_loaded_pointer():
    loaded: set[str] = set()
    reg = _reg({"a": Skill("a", "d", "FULL BODY")}, loaded)
    first = reg.use_skill("a")
    second = reg.use_skill("a")
    assert "FULL BODY" in first
    assert "FULL BODY" not in second  # body not re-injected
    assert "already loaded" in second.lower()


def test_use_skill_unknown_name_lists_available():
    reg = _reg({"alpha": Skill("alpha", "d"), "beta": Skill("beta", "d")})
    out = reg.use_skill("gamma")
    assert out.startswith("Error:")
    assert "alpha" in out
    assert "beta" in out


def test_use_skill_unknown_with_no_skills():
    reg = _reg({})
    out = reg.use_skill("x")
    assert out.startswith("Error:")
    assert "none available" in out


def test_use_skill_oversized_body_offloads():
    big = "x" * (ARTIFACT_THRESHOLD + 100)
    reg = _reg({"big": Skill("big", "d", big)})
    out = reg.use_skill("big")
    # Same artifact contract as a large read_file: (None, {"stdout":..., "ok":True}).
    assert isinstance(out, tuple)
    assert out[0] is None
    assert out[1]["ok"] is True
    assert big in out[1]["stdout"]


# ── concurrency: use_skill is NOT parallel-safe (mutates loaded-set) ───────────


def test_use_skill_not_parallel_safe():
    assert is_parallel_safe("use_skill", {"name": "a"}) is False


# ── conditional registration: tool schemas byte-identical when no skills ───────


def test_tool_schemas_byte_identical_without_skills():
    from my_coding_agent.cli import _all_tools, _build_tools

    assert _build_tools({}) == _all_tools()
    assert "use_skill" not in {t["function"]["name"] for t in _build_tools({})}


def test_use_skill_registered_only_with_skills():
    from my_coding_agent.cli import _build_tools

    tools = _build_tools({"a": Skill("a", "does a")})
    assert "use_skill" in {t["function"]["name"] for t in tools}


def test_system_prompt_unaffected_by_skills():
    # Skills never touch the system prompt (#75 invariant). _system_prompt takes
    # no skill input, so it is identical regardless of what is on disk.
    from my_coding_agent.cli import _system_prompt

    prompt = _system_prompt()
    assert "use_skill" not in prompt
    assert "Available skills" not in prompt

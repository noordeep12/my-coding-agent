"""Tests for the ``use_skill`` tool: lazy load, dedup, error, offload, routing."""

from my_coding_agent.engine.routing import _BASELINE_TOOLS, ToolRouter
from my_coding_agent.engine.tool_execution.concurrency import is_parallel_safe
from my_coding_agent.engine.tool_registry import ARTIFACT_THRESHOLD
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.engine.tool_registry.skills import Skill


def _reg(skills, loaded=None):
    return ToolsRegistry(
        skills=skills, loaded_skills=loaded if loaded is not None else set()
    )


def _tool_def(name, tags=None):
    return {"type": "function", "function": {"name": name}, "tags": tags or []}


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
    assert "alpha" in out and "beta" in out


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


# ── routing baseline ──────────────────────────────────────────────────────────


class _NoLLM:
    def chat_completion(self, *a, **k):  # pragma: no cover - never called here
        raise AssertionError("router should not call the LLM in these cases")


def test_baseline_contains_use_skill_when_registered():
    router = ToolRouter(_NoLLM())
    tools = [
        _tool_def("bash"),
        _tool_def("read_file"),
        _tool_def("read_tool_artifact"),
        _tool_def("use_skill"),
        _tool_def("delegate", tags=["delegate"]),
    ]
    # A keyword match on a non-baseline tool (delegate) still keeps the whole
    # baseline — including use_skill — in the selection (tool-routing).
    selected, phase = router.route_tools("please delegate this", tools)
    names = {t["function"]["name"] for t in selected}
    assert "use_skill" in names
    assert "delegate" in names


def test_baseline_excludes_use_skill_when_absent():
    # With no use_skill in the toolset, the effective baseline is exactly today's.
    router = ToolRouter(_NoLLM())
    tools = [
        _tool_def("bash"),
        _tool_def("read_file"),
        _tool_def("read_tool_artifact"),
        _tool_def("read_article", tags=["web"]),
    ]
    selected, _ = router.route_tools("fetch a web page", tools)
    names = {t["function"]["name"] for t in selected}
    assert "use_skill" not in names
    assert {"bash", "read_file", "read_tool_artifact"} <= names


def test_baseline_constant_includes_use_skill():
    assert "use_skill" in _BASELINE_TOOLS


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

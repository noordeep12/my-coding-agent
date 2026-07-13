"""Skill discovery, frontmatter parsing, and index rendering (issue #19)."""

from pathlib import Path

from my_coding_agent.engine.tool_execution.schema import SKILL_INDEX_TOTAL_MAX_CHARS
from my_coding_agent.engine.tool_registry.skills import (
    TIER_FULL,
    TIER_NAMES_ONLY,
    TIER_NONE,
    TIER_TRUNCATED,
    Skill,
    _parse_frontmatter,
    build_opening_block,
    discover_skills,
    render_skill_index,
)


def _write_skill(
    root: Path, dir_name: str, name: str, description: str, body: str = ""
) -> None:
    skill_dir = root / ".my_coding_agent" / "skills" / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}"
    )


# ── frontmatter parsing ───────────────────────────────────────────────────────


def test_parse_frontmatter_basic():
    fm, body = _parse_frontmatter("---\nname: a\ndescription: does a\n---\nBody text")
    assert fm == {"name": "a", "description": "does a"}
    assert body == "Body text"


def test_parse_frontmatter_ignores_unknown_keys():
    fm, _ = _parse_frontmatter(
        "---\nname: a\ndescription: d\nversion: 3\nauthor: x\n---\nb"
    )
    assert fm["name"] == "a"
    assert fm["description"] == "d"
    assert fm["version"] == "3"  # kept but harmless; builder ignores extras


def test_parse_frontmatter_strips_quotes():
    fm, _ = _parse_frontmatter("---\nname: \"a\"\ndescription: 'd'\n---\nb")
    assert fm == {"name": "a", "description": "d"}


def test_parse_frontmatter_no_opening_fence_is_malformed():
    assert _parse_frontmatter("name: a\ndescription: d\nBody") is None


def test_parse_frontmatter_no_closing_fence_is_malformed():
    assert _parse_frontmatter("---\nname: a\ndescription: d\nno closing") is None


def test_parse_frontmatter_line_without_colon_is_malformed():
    assert _parse_frontmatter("---\nname a\n---\nb") is None


# ── discovery ─────────────────────────────────────────────────────────────────


def test_discover_empty_roots(tmp_path):
    assert discover_skills(project_root=tmp_path, user_root=tmp_path / "home") == {}


def test_discover_single_skill(tmp_path):
    _write_skill(tmp_path, "greet", "greet", "greet the user", "say hi")
    skills = discover_skills(project_root=tmp_path, user_root=tmp_path / "home")
    assert set(skills) == {"greet"}
    assert skills["greet"].description == "greet the user"
    assert skills["greet"].body == "say hi"


def test_project_shadows_user(tmp_path):
    home = tmp_path / "home"
    _write_skill(home, "dep", "deploy", "USER version", "user body")
    _write_skill(tmp_path, "dep", "deploy", "PROJECT version", "project body")
    skills = discover_skills(project_root=tmp_path, user_root=home)
    assert skills["deploy"].description == "PROJECT version"
    assert skills["deploy"].body == "project body"


def test_discover_deterministic_sorted_order(tmp_path):
    for n in ["zebra", "alpha", "mango"]:
        _write_skill(tmp_path, n, n, f"{n} desc")
    skills = discover_skills(project_root=tmp_path, user_root=tmp_path / "home")
    assert list(skills) == ["alpha", "mango", "zebra"]


def test_malformed_skill_skipped_one_warning(tmp_path, caplog):
    _write_skill(tmp_path, "good", "good", "good desc")
    bad_dir = tmp_path / ".my_coding_agent" / "skills" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text("no frontmatter here")
    import logging

    with caplog.at_level(logging.WARNING):
        skills = discover_skills(project_root=tmp_path, user_root=tmp_path / "home")
    assert set(skills) == {"good"}  # run proceeds, bad skill dropped
    warnings = [r for r in caplog.records if "bad" in r.getMessage().lower()]
    assert len(warnings) == 1


def test_missing_required_fields_skipped(tmp_path):
    d = tmp_path / ".my_coding_agent" / "skills" / "partial"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: partial\n---\nbody")  # no description
    skills = discover_skills(project_root=tmp_path, user_root=tmp_path / "home")
    assert skills == {}


def test_directory_without_skill_md_ignored(tmp_path):
    (tmp_path / ".my_coding_agent" / "skills" / "empty").mkdir(parents=True)
    assert discover_skills(project_root=tmp_path, user_root=tmp_path / "home") == {}


# ── index rendering + budget ──────────────────────────────────────────────────


def test_render_empty_index():
    idx = render_skill_index({})
    assert idx.text == ""
    assert idx.names == []
    assert idx.tier == TIER_NONE


def test_render_full_tier_lists_all_skills():
    skills = {
        "a": Skill("a", "does a"),
        "b": Skill("b", "does b"),
    }
    idx = render_skill_index(skills)
    assert idx.tier == TIER_FULL
    assert "use_skill" in idx.text
    assert "- a: does a" in idx.text
    assert "- b: does b" in idx.text
    assert idx.names == ["a", "b"]
    assert idx.chars == len(idx.text)


def test_render_header_states_precedence_and_safety():
    idx = render_skill_index({"a": Skill("a", "d")})
    assert "shadows" in idx.text  # project-shadows-user precedence
    assert "never override the safety rules" in idx.text  # D8 precedence sentence


def test_per_entry_truncation_of_long_description():
    long_desc = "x" * 500
    idx = render_skill_index({"a": Skill("a", long_desc)})
    line = [ln for ln in idx.text.splitlines() if ln.startswith("- a:")][0]
    assert len(line) <= 200
    assert line.endswith("…")


def test_over_budget_degrades_to_truncated_then_names_only():
    # Many skills with long descriptions force degradation past the full tier.
    many = {f"s{i:02d}": Skill(f"s{i:02d}", "d" * 180) for i in range(40)}
    idx = render_skill_index(many)
    assert idx.tier in (TIER_TRUNCATED, TIER_NAMES_ONLY)
    assert len(idx.text) <= SKILL_INDEX_TOTAL_MAX_CHARS


def test_names_only_tier_never_exceeds_cap():
    # A very large number of skills forces the names-only tier.
    many = {f"skill{i:03d}": Skill(f"skill{i:03d}", "d" * 190) for i in range(300)}
    idx = render_skill_index(many)
    assert idx.tier == TIER_NAMES_ONLY
    assert len(idx.text) <= SKILL_INDEX_TOTAL_MAX_CHARS


def test_render_always_within_total_cap():
    many = {f"s{i:03d}": Skill(f"s{i:03d}", "long " * 50) for i in range(100)}
    idx = render_skill_index(many)
    assert len(idx.text) <= SKILL_INDEX_TOTAL_MAX_CHARS


# ── opening block placement helper ────────────────────────────────────────────


def test_build_opening_block_empty_when_no_skills():
    block, idx = build_opening_block({})
    assert block == ""
    assert idx.tier == TIER_NONE


def test_build_opening_block_injects_loaded_bodies():
    skills = {"a": Skill("a", "does a", "FULL BODY A"), "b": Skill("b", "does b", "B")}
    block, placed = build_opening_block(skills, loaded_names={"a"})
    assert "FULL BODY A" in block  # loaded skill body re-injected (D6)
    assert "Skill: a" in block
    assert "FULL BODY" not in "".join(  # unloaded skill 'b' stays index-only
        ln for ln in block.splitlines() if "does b" in ln
    )
    assert placed.chars == len(block)
    assert placed.chars > render_skill_index(skills).chars  # bodies add size

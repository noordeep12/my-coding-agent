"""Skill discovery, parsing, and index rendering (skill-knowledge-delivery, #19).

A *skill* is a directory holding a ``SKILL.md`` file with minimal YAML-ish
frontmatter (``name`` and ``description``) and a free-Markdown body of procedural
knowledge. Skills let a user steer the agent's tool usage without editing Python
source: at session start the discovered skills are rendered into a compact index
appended to the opening user message, and the agent loads a skill's full body on
demand with the ``use_skill`` tool.

This module is deliberately dependency-free of any YAML library — the frontmatter
is two string fields, split by a small hand-rolled parser. Malformed frontmatter
never fails a run; the offending skill is skipped with one warning.

Discovery scans ``<project>/.my_coding_agent/skills/*/SKILL.md`` and
``~/.my_coding_agent/skills/*/SKILL.md`` exactly once (a stable per-run snapshot);
a project skill shadows a same-named user skill, and the result is sorted by name
for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...utils import get_logger
from ..tool_execution.schema import (
    SKILL_INDEX_PER_ENTRY_MAX_CHARS,
    SKILL_INDEX_TOTAL_MAX_CHARS,
)

logger = get_logger(__name__)

# Discovery locations (D2): project then user, both under ``.my_coding_agent/skills``.
_SKILLS_SUBPATH = Path(".my_coding_agent") / "skills"

# Fixed index header (D3 + D8): names the loading tool, states the project-shadows-
# user precedence, and carries the single precedence sentence (skills guide tool
# usage but never override safety rules). This block is appended to the opening
# user message, never the system prompt, so the #75 prefix-cache invariant holds.
_INDEX_HEADER = (
    "## Available skills\n"
    "These skills hold procedural knowledge for specific tasks. When a task "
    "matches one, call use_skill(name) to load its full instructions before "
    "acting; a project skill shadows a user skill of the same name. Skills "
    "guide how to drive the tools but never override the safety rules above."
)

# Degradation tiers reported on the skill-index observability event (D4/D9).
TIER_NONE = "none"
TIER_FULL = "full"
TIER_TRUNCATED = "truncated"
TIER_NAMES_ONLY = "names_only"


@dataclass(frozen=True)
class Skill:
    """One discovered skill: its frontmatter identity plus its Markdown body.

    ``name`` and ``description`` come from the frontmatter (``description``
    doubles as the when-to-use hint shown in the index); ``body`` is the full
    Markdown loaded on demand by ``use_skill``.
    """

    name: str
    description: str
    body: str = ""


@dataclass(frozen=True)
class RenderedIndex:
    """The rendered skill-index block plus the metadata the recorder captures.

    ``text`` is what gets appended to the opening user message (empty when there
    are no skills); ``names`` lists the skills included; ``chars`` is the block
    size actually placed; ``tier`` is the degradation tier applied (D4).
    """

    text: str
    names: list[str]
    chars: int
    tier: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str] | None:
    """Split a ``SKILL.md`` into (frontmatter, body), or None if malformed.

    The frontmatter is the block between a leading ``---`` line and the next
    ``---`` line; each non-blank line inside is a ``key: value`` pair (surrounding
    quotes on the value are stripped). Returns None when there is no opening
    fence, no closing fence, or a non-blank frontmatter line without a colon —
    all treated as malformed so the caller skips the skill with one warning.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    frontmatter: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            return None
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    body = "\n".join(lines[end + 1 :]).strip()
    return frontmatter, body


def _build_skill(md_path: Path) -> Skill | None:
    """Parse one ``SKILL.md`` into a ``Skill``, or None when it must be skipped.

    Unknown frontmatter keys are ignored (forward-compatible). A skill is skipped
    (one warning, never a failed run) when the file is unreadable, the frontmatter
    is malformed, or the required ``name``/``description`` fields are missing.
    """
    try:
        text = md_path.read_text()
    except OSError as exc:
        logger.warning("Skipping unreadable skill file %s: %s", md_path, exc)
        return None
    parsed = _parse_frontmatter(text)
    if parsed is None:
        logger.warning("Skipping skill with malformed frontmatter: %s", md_path)
        return None
    frontmatter, body = parsed
    name = frontmatter.get("name", "").strip()
    description = frontmatter.get("description", "").strip()
    if not name or not description:
        logger.warning(
            "Skipping skill missing 'name'/'description' frontmatter: %s", md_path
        )
        return None
    return Skill(name=name, description=description, body=body)


def _scan_root(root: Path) -> list[Skill]:
    """Return every valid skill under ``root/*/SKILL.md``, in sorted directory order."""
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        md_path = skill_dir / "SKILL.md"
        if not md_path.is_file():
            continue
        skill = _build_skill(md_path)
        if skill is not None:
            skills.append(skill)
    return skills


def discover_skills(
    project_root: str | Path | None = None,
    user_root: str | Path | None = None,
) -> dict[str, Skill]:
    """Discover skills once (D2): project then user, project shadows user.

    Scans ``<project_root>/.my_coding_agent/skills`` and
    ``<user_root>/.my_coding_agent/skills`` (defaulting to the cwd and the home
    directory). A project skill overrides a same-named user skill. The returned
    mapping is sorted by name so the snapshot — and every index rendered from it
    — is deterministic. Empty roots yield an empty mapping (no index anywhere).

    Args:
        project_root: Project directory to scan (defaults to the cwd).
        user_root: User home directory to scan (defaults to ``Path.home()``).

    Returns:
        A name → :class:`Skill` mapping, sorted by name.
    """
    project = Path(project_root) if project_root is not None else Path.cwd()
    user = Path(user_root) if user_root is not None else Path.home()
    merged: dict[str, Skill] = {}
    for skill in _scan_root(user / _SKILLS_SUBPATH):
        merged[skill.name] = skill
    for skill in _scan_root(project / _SKILLS_SUBPATH):
        merged[skill.name] = skill  # project shadows user
    return dict(sorted(merged.items()))


def _truncate(text: str, cap: int) -> str:
    """Truncate ``text`` to ``cap`` chars, marking a cut with a trailing ellipsis."""
    if len(text) <= cap:
        return text
    if cap <= 1:
        return "…"
    return text[: cap - 1].rstrip() + "…"


def _entry_line(skill: Skill, line_cap: int) -> str:
    """Render one index line, truncating the description to fit ``line_cap``."""
    prefix = f"- {skill.name}: "
    if len(prefix) >= line_cap:
        return _truncate(f"- {skill.name}", line_cap)
    return prefix + _truncate(skill.description, line_cap - len(prefix))


def _names_only_lines(skills: list[Skill], budget: int) -> list[str]:
    """Render names-only lines that together stay within ``budget`` chars."""
    lines: list[str] = []
    used = 0
    for skill in skills:
        line = f"- {skill.name}"
        used += len(line) + 1  # +1 for the joining newline
        if used > budget:
            break
        lines.append(line)
    return lines


def render_skill_index(skills: dict[str, Skill]) -> RenderedIndex:
    """Render the skill index deterministically within the fixed budget (D4).

    Three degradation tiers, applied in order and never exceeding the total cap:

    - ``full`` — one ``- name: description`` line per skill, each capped at the
      per-entry limit, when the whole block fits the total cap.
    - ``truncated`` — descriptions truncated to an even per-line share so the
      block fits.
    - ``names_only`` — one ``- name`` line per skill when even truncated
      descriptions overflow.

    Returns an empty :class:`RenderedIndex` (tier ``none``) when there are no
    skills, so the opening message is left byte-identical to today.
    """
    if not skills:
        return RenderedIndex(text="", names=[], chars=0, tier=TIER_NONE)
    names = list(skills)
    skill_list = [skills[n] for n in names]

    full_lines = [_entry_line(s, SKILL_INDEX_PER_ENTRY_MAX_CHARS) for s in skill_list]
    text = _INDEX_HEADER + "\n" + "\n".join(full_lines)
    if len(text) <= SKILL_INDEX_TOTAL_MAX_CHARS:
        return RenderedIndex(text, names, len(text), TIER_FULL)

    # Tier 2 — truncate descriptions to an even per-line share of the remaining
    # budget (reserve one newline per line plus the header newline).
    budget = SKILL_INDEX_TOTAL_MAX_CHARS - len(_INDEX_HEADER) - len(skill_list) - 1
    per_line = budget // len(skill_list)
    if per_line >= len(f"- {min(names, key=len)}: ") + 1:
        trunc_lines = [_entry_line(s, per_line) for s in skill_list]
        text = _INDEX_HEADER + "\n" + "\n".join(trunc_lines)
        if len(text) <= SKILL_INDEX_TOTAL_MAX_CHARS:
            return RenderedIndex(text, names, len(text), TIER_TRUNCATED)

    # Tier 3 — names only, dropping trailing entries if even that overflows.
    names_budget = SKILL_INDEX_TOTAL_MAX_CHARS - len(_INDEX_HEADER) - 1
    lines = _names_only_lines(skill_list, names_budget)
    text = _INDEX_HEADER + "\n" + "\n".join(lines)
    return RenderedIndex(text, names, len(text), TIER_NAMES_ONLY)


def render_loaded_bodies(skills: dict[str, Skill], loaded_names: set[str]) -> str:
    """Render the full body of each already-loaded skill for a continuation (D6).

    Each body is prefixed with a ``Skill: <name>`` header (matching ``use_skill``
    output) so a continuation reads exactly what the pre-reset run saw. Names not
    present in the current snapshot are skipped. Order is deterministic (sorted).
    """
    parts: list[str] = []
    for name in sorted(loaded_names):
        skill = skills.get(name)
        if skill is not None:
            parts.append(f"Skill: {name}\n\n{skill.body}")
    return "\n\n".join(parts)


def build_opening_block(
    skills: dict[str, Skill], loaded_names: set[str] | None = None
) -> tuple[str, RenderedIndex]:
    """Build the block to append to a session's opening user message.

    The block is the rendered index (D3) followed, for a continuation, by the
    full body of each previously-loaded skill (D6). Returns ``("", ...)`` when
    there are no skills so callers append nothing. The returned
    :class:`RenderedIndex` reflects what was actually placed — its ``chars``
    include any re-injected bodies — so the recorder's skill-index event mirrors
    the continuation's real payload (D9).

    Args:
        skills: The discovered-skill snapshot for this run.
        loaded_names: Names of skills already loaded before a context reset;
            their full bodies are re-injected after the index. ``None`` (a fresh
            run) injects no bodies.

    Returns:
        ``(block_text, placed_index)``.
    """
    index = render_skill_index(skills)
    if not index.text:
        return "", index
    block = index.text
    if loaded_names:
        bodies = render_loaded_bodies(skills, loaded_names)
        if bodies:
            block = f"{block}\n\n{bodies}"
    placed = RenderedIndex(
        text=block, names=index.names, chars=len(block), tier=index.tier
    )
    return block, placed

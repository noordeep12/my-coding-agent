"""Untrusted-content confinement: provenance marking, demarcation, and the
first-increment capability reduction (issue #128).

The agent ingests external content — web pages via ``fetch_web``, and (in
later increments) pulled repo/file content — and without confinement treats
the words in it with the same authority as the developer's task. This module
marks such content untrusted at entry, demarcates it as data (not
instructions) to the model, and reduces the action surface while it is
active: build/install-script execution from a freshly-cloned external repo
is not auto-run at full privilege.

State (``is_untrusted_active`` / ``is_freshly_cloned``) is run-scoped via
``contextvars`` — set once ingestion/cloning is observed and never cleared —
so once untrusted content or a clone enters a run, the reduced posture holds
for the rest of that run. Deliberately simple: a full taint-tracking or
lifecycle model is a known limitation (design.md decision 2 / risks).
"""

from __future__ import annotations

import contextvars
import re

from .schema import DEMARCATION_CLOSE, DEMARCATION_OPEN, TRUSTED, UNTRUSTED, Reduction

__all__ = [
    "TRUSTED",
    "UNTRUSTED",
    "Reduction",
    "mark_untrusted",
    "demarcate",
    "note_untrusted_content",
    "is_untrusted_active",
    "note_bash_command",
    "is_freshly_cloned",
    "check_reduction",
]

# Set the first time this run tags a tool result untrusted; never cleared.
_untrusted_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "untrusted_active", default=False
)
# Set the first time this run observes a successful `git clone`; never cleared.
_freshly_cloned: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "freshly_cloned", default=False
)


def mark_untrusted(metadata: dict) -> dict:
    """Return a copy of ``metadata`` with the untrusted-provenance marker set."""
    tagged = dict(metadata or {})
    tagged["provenance"] = UNTRUSTED
    return tagged


def demarcate(content: str) -> str:
    """Wrap untrusted content in explicit data-not-instructions framing."""
    return f"{DEMARCATION_OPEN}{content}{DEMARCATION_CLOSE}"


def note_untrusted_content() -> None:
    """Record that untrusted ingested content has entered this run's context."""
    _untrusted_active.set(True)


def is_untrusted_active() -> bool:
    """Whether untrusted ingested content is active in this run."""
    return _untrusted_active.get()


_GIT_CLONE_RE = re.compile(r"\bgit\s+clone\b")


def note_bash_command(command: str, ok: bool) -> None:
    """Update freshly-cloned-repo state from one completed ``bash`` call."""
    if ok and _GIT_CLONE_RE.search(command):
        _freshly_cloned.set(True)


def is_freshly_cloned() -> bool:
    """Whether this run has cloned an external repository."""
    return _freshly_cloned.get()


# Build/install/setup commands the clone-and-build reduction applies to —
# narrow and high-signal, the same bias as `tool_execution.policy`: a false
# positive here blocks legitimate work, so this only matches the concrete
# clone-and-build pattern (design.md decision 3 / the 0DIN incident).
_INSTALL_BUILD_RE = re.compile(
    r"\b(npm\s+(install|ci)|yarn\s+install|pip3?\s+install|make\b|"
    r"\./configure\b|\bsh\s+\S+\.sh\b|\bbash\s+\S+\.sh\b|\./\S+\.sh\b|"
    r"cargo\s+build|python3?\s+setup\.py)"
)


def check_reduction(func_name: str, args: dict) -> Reduction | None:
    """Return a :class:`Reduction` when this call is subject to the
    clone-and-build capability reduction, else ``None``.

    Fires only when the run has both cloned an external repo and has
    untrusted content active, and the command matches a build/install/setup
    pattern — the concrete threat is a freshly-cloned repo's own build
    instructions steering execution (the 0DIN incident). No sandbox (#25)
    exists yet to route through, so the first increment refuses outright.
    """
    if func_name != "bash":
        return None
    if not (is_untrusted_active() and is_freshly_cloned()):
        return None
    command = args.get("command", "")
    if not command or not _INSTALL_BUILD_RE.search(command):
        return None
    return Reduction(
        rule_id="clone_and_build_untrusted",
        reason=(
            "Build/install/setup-script execution from a freshly-cloned "
            "external repository, while untrusted ingested content is "
            "active, is not auto-run at full privilege (no sandbox "
            "available for this run)."
        ),
        safer_alternative=(
            "Read the script's contents first and confirm it does nothing "
            "unexpected, then run it explicitly with that reviewed content "
            "in mind."
        ),
    )

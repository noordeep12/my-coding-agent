"""Dataclasses and constants for the bash OS-level sandbox (issue #25).

Kept separate from ``__init__.py`` so the data contract (scope, host
capability) can be imported without pulling in the profile-generation logic.
Stdlib-only leaf, like ``tool_execution/policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Opt-in switch (off by default): set to any value other than ""/"0"/"false" to
# enable the sandbox for the process. Read at call time (not import time), so
# the CLI's ``--sandbox`` flag — which just sets this var before the first tool
# call — and a shell-exported var behave identically. Being an environment
# variable (not a RunContext field) means a ``delegate``-spawned subagent, which
# runs in the same process, inherits the exact same sandbox posture for free —
# the same propagation mechanism already used by the dangerous-command refusal
# gate's ``MCA_DISABLE_DANGEROUS_COMMAND_GATE``.
ENV_VAR = "MCA_BASH_SANDBOX"


@dataclass(frozen=True)
class SandboxScope:
    """The permitted filesystem write scope for one sandboxed run.

    Fixed at startup from the workspace root plus a minimal temp/cache
    allowlist — never expanded by anything in the command text.
    """

    workspace_root: Path
    extra_write_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class HostCapability:
    """Whether this host can run the sandbox mechanism, and why not if not."""

    supported: bool
    reason: str

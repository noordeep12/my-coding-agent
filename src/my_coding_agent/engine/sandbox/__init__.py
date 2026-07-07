"""OS-level sandbox for ``bash`` subprocesses on macOS (issue #25).

Wraps a ``bash`` command in Apple's Seatbelt sandbox (``sandbox-exec`` with a
generated profile) so the confinement is enforced by the OS beneath the shell,
independent of how the command text is phrased or obfuscated — the same
mechanism Claude Code's sandbox-runtime and OpenAI Codex ship on macOS. Default
deny: filesystem writes are confined to the workspace root plus a small
temp/cache allowlist, and all outbound network is denied (a separate egress
allow/deny policy, #126, is expected to layer on top of this boundary later).

Opt-in via :data:`schema.ENV_VAR`, gated CLI-side (``--sandbox``); off means
``registry.bash`` runs exactly as it did before this module existed. On, but
unsupported (non-macOS, or ``sandbox-exec`` missing), the caller must refuse to
run unconfined rather than silently falling back — see :func:`probe_host_capability`.

Stdlib + subprocess only, no internal imports — a leaf module like
``tool_execution/policy.py``.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from .schema import ENV_VAR, HostCapability, SandboxScope

__all__ = [
    "ENV_VAR",
    "HostCapability",
    "SandboxScope",
    "is_enabled",
    "probe_host_capability",
    "default_scope",
    "build_profile",
    "wrap_command",
    "is_likely_denial",
]


def is_enabled() -> bool:
    """Return ``True`` when the bash sandbox is enabled for this process."""
    raw = os.environ.get(ENV_VAR, "")
    return raw.strip().lower() not in ("", "0", "false")


def probe_host_capability() -> HostCapability:
    """Return whether this host can run the sandbox mechanism, and why not."""
    system = platform.system()
    if system != "Darwin":
        return HostCapability(
            supported=False,
            reason=f"unsupported platform {system!r}: the sandbox requires macOS",
        )
    if shutil.which("sandbox-exec") is None:
        return HostCapability(
            supported=False,
            reason="'sandbox-exec' not found on PATH",
        )
    return HostCapability(supported=True, reason="")


# Temp/cache directories common tooling (pip, npm, pytest, editors) writes to
# that are not under the workspace root itself. Deliberately narrow: macOS's
# per-user temp root (the parent of ``$TMPDIR``, e.g. ``/private/var/folders``)
# is NOT included — it is shared machine-wide across every process's own temp
# subdirectory, so allowing the whole parent would let a sandboxed command
# write into any other process's temp space, defeating the scope. Only the
# process's own ``$TMPDIR`` (specific, e.g. ``/var/folders/<hash>/<hash>/T``)
# and the classic world-writable ``/tmp`` are allowed.
# nosec B108 — these are sandbox *allowlist* entries the profile permits
# writes to, not a location this module itself writes a file to; the
# hardcoded-tmp-dir check doesn't apply to this usage.
_TEMP_ALLOWLIST_CANDIDATES = ("/tmp", "/private/tmp")  # nosec B108


def default_scope(workspace_root: Path | str) -> SandboxScope:
    """Build the permitted scope from the workspace root plus a temp allowlist.

    Resolved once, before the run — the scope never expands based on anything
    in a model-generated command.
    """
    root = Path(workspace_root).resolve()
    candidates = [*_TEMP_ALLOWLIST_CANDIDATES, os.environ.get("TMPDIR", "")]
    temp_paths = tuple(
        dict.fromkeys(Path(p).resolve() for p in candidates if p and Path(p).exists())
    )
    return SandboxScope(workspace_root=root, extra_write_paths=temp_paths)


_PROFILE_TEMPLATE = """(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow file-read*)
{write_rules}
(deny network*)
"""


def build_profile(scope: SandboxScope) -> str:
    """Generate the Seatbelt profile text for ``scope``.

    Default-deny for everything; reads are allowed broadly (tooling needs to
    read the system and its own dependencies — read-scope tightening for
    secret files is deferred, see design.md), writes are allowed only under
    the workspace root and the temp allowlist, and all network is denied.
    """
    write_roots = (scope.workspace_root, *scope.extra_write_paths)
    write_rules = "\n".join(
        f'(allow file-write* (subpath "{root}"))' for root in write_roots
    )
    return _PROFILE_TEMPLATE.format(write_rules=write_rules)


def wrap_command(command: str, scope: SandboxScope) -> list[str]:
    """Return the ``sandbox-exec`` argv that runs ``command`` inside ``scope``.

    The command still runs through a shell (``/bin/sh -c``) so pipes,
    redirections, and builtins keep working exactly as the unsandboxed path
    supports them — only the outer process is now ``sandbox-exec``, applying
    to the whole process tree the shell spawns.
    """
    profile = build_profile(scope)
    return ["sandbox-exec", "-p", profile, "/bin/sh", "-c", command]


# Seatbelt-denied syscalls surface to the calling tool as EPERM, whose libc
# message is "Operation not permitted" — distinct from the "Permission denied"
# (EACCES) text ordinary Unix permission failures produce, so this is a
# reasonably specific (not perfect) signal that a failure was the sandbox
# denying an operation rather than the command failing on its own terms.
_DENIAL_MARKER = "operation not permitted"


def is_likely_denial(stderr: str) -> bool:
    """Best-effort classification of a failed, sandboxed command's stderr.

    Purely for observability (:meth:`Recorder.record_sandbox_denial`) — never
    used to decide enforcement, which the OS already did unconditionally.
    """
    return _DENIAL_MARKER in stderr.lower()

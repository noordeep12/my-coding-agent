"""Dangerous-command refusal policy — deterministic, local, pre-execution.

This module is a stdlib-only leaf (schema-adjacent, like ``concurrency.py``):
no internal imports, so it can be depended on from anywhere without a cycle.
Rules are data (:class:`Rule`), each matched against a ``bash`` command line by
a plain predicate. :func:`evaluate` is the single entry point the executor
calls before dispatch — it never executes anything and never calls the
recorder; enforcement here stays observability-free (see design.md decision 5).

Deliberately narrow and high-signal, the opposite bias of
``concurrency.is_parallel_safe``: a false positive here blocks legitimate work
and erodes trust in the gate, so a rule only fires on unambiguous danger. Not
exhaustive and not a sandbox — obfuscation (base64, `$IFS`, `eval`) defeats a
textual gate by design; this is a last line of defense on top of the advisory
system prompt, not containment of a determined adversary.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """One recognized security-standard/best-practice citation."""

    standard_id: str
    url: str


@dataclass(frozen=True)
class Refusal:
    """The decision returned when a command matches a :class:`Rule`."""

    rule_id: str
    reason: str
    references: tuple[Reference, ...]
    safer_alternative: str


@dataclass(frozen=True)
class Rule:
    """One dangerous-operation rule: a predicate plus its explanation."""

    rule_id: str
    predicate: Callable[[str], bool]
    reason: str
    references: tuple[Reference, ...]
    safer_alternative: str

    def to_refusal(self) -> Refusal:
        return Refusal(
            rule_id=self.rule_id,
            reason=self.reason,
            references=self.references,
            safer_alternative=self.safer_alternative,
        )


# ── predicates ─────────────────────────────────────────────────────────────
# Each predicate takes the raw command line and returns True only when it is
# confident the command matches the dangerous pattern. Matching normalizes
# whitespace but does not attempt de-obfuscation (documented limitation).

# Recursive force-delete of a root/home-class path: `rm -rf /`, `rm -rf ~`,
# `rm -rf /*`, `rm -rf $HOME`, `rm -fr /Users/x`, etc. A close-but-safe
# look-alike like `rm -rf ./build` or `rm -rf build/` must NOT fire.
_RM_ROOT_RE = re.compile(
    r"""
    \brm\s+
    (?:-[a-zA-Z]*[rf][a-zA-Z]*\s+|--recursive\s+|--force\s+)*   # flag(s)
    (?:-[a-zA-Z]*[rf][a-zA-Z]*\s+|--recursive\s+|--force\s+)    # need >=1 r/f flag
    (?:-[a-zA-Z]*[rf][a-zA-Z]*\s+|--recursive\s+|--force\s+)*   # more flags, any order
    (["']?)
    (?:/\*?|~|\$HOME|/(?:Users|home)/[^/\s]+/?|/root/?)
    \1(?:\s|$)
    """,
    re.VERBOSE,
)


def _is_rm_root(command: str) -> bool:
    return bool(_RM_ROOT_RE.search(command.strip()))


# Remote content piped into a shell interpreter: `curl ... | sh`,
# `wget -O- ... | bash`, `curl ... | sudo bash`, etc.
_FETCHERS = ("curl", "wget", "fetch")
_SHELLS = ("sh", "bash", "zsh", "dash", "ksh")


def _is_remote_pipe_to_shell(command: str) -> bool:
    stages = [s.strip() for s in command.split("|")]
    if len(stages) < 2:
        return False
    try:
        first_words = shlex.split(stages[0])
    except ValueError:
        return False
    if not first_words or first_words[0] not in _FETCHERS:
        return False
    last_stage = stages[-1]
    try:
        last_words = shlex.split(last_stage)
    except ValueError:
        return False
    # Skip a leading `sudo` (or similar) when checking the shell name.
    words = [w for w in last_words if w not in ("sudo",)]
    if not words:
        return False
    shell_name = words[0].rsplit("/", 1)[-1]
    return shell_name in _SHELLS


# Raw-device writes: `dd of=/dev/...`, `mkfs...` against a device.
_DD_OF_DEV_RE = re.compile(r"\bdd\b[^|;]*\bof=/dev/")
_MKFS_RE = re.compile(r"\bmkfs(?:\.\w+)?\s")


def _is_raw_device_write(command: str) -> bool:
    return bool(_DD_OF_DEV_RE.search(command)) or bool(_MKFS_RE.search(command))


# Classic shell fork bomb: `:(){ :|:& };:` (and cosmetic variants).
_FORK_BOMB_RE = re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&?\s*\}\s*;\s*:")


def _is_fork_bomb(command: str) -> bool:
    return bool(_FORK_BOMB_RE.search(command))


# World-writable permission blast on a system path: `chmod -R 777 /`,
# `chmod 777 /etc`, `chmod -R 777 /usr`, etc.
_CHMOD_BLAST_RE = re.compile(
    r"\bchmod\s+(?:-R\s+)?0?777\s+(/(?:etc|usr|bin|sbin|var|System|Library)\b|/)"
)


def _is_permission_blast(command: str) -> bool:
    return bool(_CHMOD_BLAST_RE.search(command))


# Credential-file exfiltration over the network: reading a well-known
# credential path and piping/passing it to a network tool in the same
# command line.
_CREDENTIAL_PATH_RE = re.compile(
    r"(~/\.ssh/id_\w+|~/\.aws/credentials|/etc/shadow|~/\.netrc|\.env\b)"
)
_NETWORK_TOOLS = ("curl", "wget", "nc", "ncat", "ssh", "scp", "rsync")


def _is_credential_exfiltration(command: str) -> bool:
    if not _CREDENTIAL_PATH_RE.search(command):
        return False
    try:
        words: list[str] = []
        for stage in command.split("|"):
            words.extend(shlex.split(stage))
    except ValueError:
        return False
    return any(w in _NETWORK_TOOLS for w in words)


# Destructive git history rewrite on a remote: `git push --force ...` /
# `git push --mirror ...` without `--force-with-lease`.
_GIT_FORCE_PUSH_RE = re.compile(r"\bgit\s+push\b")


def _is_destructive_git_push(command: str) -> bool:
    if not _GIT_FORCE_PUSH_RE.search(command):
        return False
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    if "--force-with-lease" in words:
        return False
    if "--mirror" in words:
        return True
    return "--force" in words or "-f" in words


# ── rule set ─────────────────────────────────────────────────────────────

_CWE_78 = Reference(
    "CWE-78", "https://cwe.mitre.org/data/definitions/78.html"
)
_OWASP_CMD_INJECTION = Reference(
    "OWASP Command Injection",
    "https://owasp.org/www-community/attacks/Command_Injection",
)
_NIST_800_53_SI_7 = Reference(
    "NIST SP 800-53 SI-7",
    "https://csrc.nist.gov/controls/sp800-53/rev5/si-7",
)
_NIST_800_53_AC_6 = Reference(
    "NIST SP 800-53 AC-6",
    "https://csrc.nist.gov/controls/sp800-53/rev5/ac-6",
)
_CWE_494 = Reference(
    "CWE-494: Download of Code Without Integrity Check",
    "https://cwe.mitre.org/data/definitions/494.html",
)
_CWE_522 = Reference(
    "CWE-522: Insufficiently Protected Credentials",
    "https://cwe.mitre.org/data/definitions/522.html",
)
_CWE_732 = Reference(
    "CWE-732: Incorrect Permission Assignment",
    "https://cwe.mitre.org/data/definitions/732.html",
)
_CWE_400 = Reference(
    "CWE-400: Uncontrolled Resource Consumption",
    "https://cwe.mitre.org/data/definitions/400.html",
)

RULES: tuple[Rule, ...] = (
    Rule(
        rule_id="rm_root_class_path",
        predicate=_is_rm_root,
        reason=(
            "Recursive force-delete targeting a root/home-class path "
            "(/, ~, $HOME) destroys the filesystem or the user's entire "
            "home directory irrecoverably."
        ),
        references=(_NIST_800_53_SI_7,),
        safer_alternative=(
            "Scope the delete to a specific, non-root subdirectory (e.g. "
            "`rm -rf ./build`), and confirm the target path exists and is "
            "the intended one before deleting."
        ),
    ),
    Rule(
        rule_id="remote_pipe_to_shell",
        predicate=_is_remote_pipe_to_shell,
        reason=(
            "Piping fetched remote content directly into a shell "
            "interpreter executes unreviewed, unauthenticated code with the "
            "user's full privileges."
        ),
        references=(_CWE_494, _OWASP_CMD_INJECTION),
        safer_alternative=(
            "Download the script to a file first, review its contents, "
            "then execute it explicitly (and verify a checksum/signature "
            "if one is published)."
        ),
    ),
    Rule(
        rule_id="raw_device_write",
        predicate=_is_raw_device_write,
        reason=(
            "Writing directly to a raw block device (`dd of=/dev/...`) or "
            "formatting one (`mkfs`) can overwrite the boot disk or destroy "
            "any filesystem on the machine."
        ),
        references=(_NIST_800_53_SI_7,),
        safer_alternative=(
            "Operate on a regular file path instead of a raw device node, "
            "or use a disk-utility command that requires an explicit, "
            "confirmed device selection outside this agent."
        ),
    ),
    Rule(
        rule_id="fork_bomb",
        predicate=_is_fork_bomb,
        reason=(
            "A fork bomb spawns processes exponentially until the machine "
            "runs out of resources, freezing or crashing it."
        ),
        references=(_CWE_400,),
        safer_alternative=(
            "Use a bounded, purposeful process (e.g. a loop with a fixed "
            "iteration count) instead of unbounded self-forking."
        ),
    ),
    Rule(
        rule_id="permission_blast",
        predicate=_is_permission_blast,
        reason=(
            "Recursively making a system path world-writable (`chmod 777`) "
            "lets any local user or process modify system files, opening a "
            "privilege-escalation path."
        ),
        references=(_CWE_732, _NIST_800_53_AC_6),
        safer_alternative=(
            "Grant the minimum permission needed to a specific file or "
            "directory (e.g. `chmod 644 <file>`), never a blanket 777 on a "
            "system path."
        ),
    ),
    Rule(
        rule_id="credential_exfiltration",
        predicate=_is_credential_exfiltration,
        reason=(
            "Reading a credential file (SSH key, cloud credentials, "
            "shadow file, .env) and passing it to a network tool in the "
            "same command line exfiltrates secrets off the machine."
        ),
        references=(_CWE_522, _OWASP_CMD_INJECTION),
        safer_alternative=(
            "Never transmit credential files over the network from this "
            "agent; use the target system's own authenticated credential-"
            "provisioning mechanism instead."
        ),
    ),
    Rule(
        rule_id="destructive_git_push",
        predicate=_is_destructive_git_push,
        reason=(
            "`git push --force`/`--mirror` overwrites the remote's history "
            "without checking for concurrent changes, permanently "
            "destroying other contributors' unmerged commits."
        ),
        references=(_NIST_800_53_SI_7,),
        safer_alternative=(
            "Use `git push --force-with-lease` instead, which aborts if the "
            "remote ref has moved since your last fetch."
        ),
    ),
)


def evaluate(func_name: str, args: dict) -> Refusal | None:
    """Return a :class:`Refusal` when this call matches a dangerous rule.

    Only ``bash`` calls are evaluated; every other tool returns ``None``
    (interface deliberately takes ``func_name`` so future non-bash rules slot
    in without a signature change). Local and deterministic: no LLM call, no
    network access, no side effect.
    """
    if func_name != "bash":
        return None
    command = args.get("command", "")
    if not command:
        return None
    for rule in RULES:
        if rule.predicate(command):
            return rule.to_refusal()
    return None

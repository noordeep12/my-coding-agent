"""Independence classification for concurrent tool dispatch.

The ``ToolExecutor`` may overlap tool calls emitted in a single assistant
message, but only when every overlapping call is provably **read-only per
call** — it cannot mutate local files, process state, or remote state. This
module is the single safety gate that decides that: :func:`is_parallel_safe`
answers "may this exact call overlap with other read-only calls?" and returns
``False`` for anything it cannot prove safe, so the executor falls back to
sequential order (the constraint: when independence cannot be established,
preserve sequential execution).

The classification is deliberately conservative — a false *negative* only costs
a missed speed-up (the call runs sequentially, exactly as before), while a false
*positive* could let a mutating call overlap and corrupt shared state. Every rule
here therefore errs toward ``False``.

Excluded on purpose even though effectively read-only:

- ``read_tool_artifact`` — makes its own LLM (extraction) call, which touches the
  recorder's single-slot ``_pending``/``_pending_child_llm_calls`` capture state;
  overlapping those is not recorder-safe, so it stays sequential.
- ``delegate`` — a subagent may perform any CRUD operation mid-run; its effects
  cannot be bounded at dispatch time (see issue #65's recorded limitation).
"""

from __future__ import annotations

import os
import shlex
from typing import Any

# Bounded worker pool for a parallel group. Kept small on purpose: the parallel
# path only ever runs I/O-bound read-only work (subprocess reads, file reads,
# HTTP GETs), so a handful of workers saturates the overlap without risking the
# machine's memory/CPU ceiling. ``MCA_TOOL_MAX_CONCURRENCY=1`` disables overlap
# entirely (every call runs sequentially, the pre-change behaviour).
_DEFAULT_MAX_CONCURRENCY = 4


def max_tool_concurrency() -> int:
    """Return the bounded worker ceiling for one parallel group (>= 1).

    Read from ``MCA_TOOL_MAX_CONCURRENCY`` at call time (not import time) so a
    test or deployment can tune it; a missing, non-integer, or < 1 value falls
    back to the conservative default, and 1 disables concurrency.
    """
    raw = os.environ.get("MCA_TOOL_MAX_CONCURRENCY")
    if raw is None:
        return _DEFAULT_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CONCURRENCY
    return value if value >= 1 else 1


# Tools whose effect is read-only for *every* argument set. ``read_tool_artifact``
# and ``delegate`` are intentionally absent (see the module docstring).
_READ_ONLY_TOOLS = frozenset({"read_file", "fetch_web"})

# Shell metacharacters that can redirect output, chain a second command, or
# substitute a command's output — any one of them means a ``bash`` command can
# no longer be proven side-effect-free by inspecting its leading words, so the
# call is treated as unsafe. (Command substitution ``$(...)`` and backticks can
# run anything; ``> < ; &`` can write files or chain writes.)
_UNSAFE_SHELL_CHARS = (">", "<", ";", "&", "`", "$(", "\n", "\r")

# Command basenames that cannot write local state or reach the network under
# *any* flag set — verified individually. Commands with an output-file flag
# (``sort -o``, ``uniq out``, ``tree -o``), an output-file positional (``xxd
# infile outfile``, ``xxd -r dump out.bin``), a command-runner form (``env
# VAR=x cmd``), a clock-set side effect (``date -s``), or network reach
# (``curl``, ``wget``) are deliberately excluded, as are the in-place editors
# ``sed``/``awk`` and ``find`` (``-exec``/``-delete``).
_READ_ONLY_COMMANDS = frozenset(
    {
        "cat", "head", "tail", "grep", "egrep", "fgrep", "rg",
        "ls", "pwd", "echo", "printf", "wc", "cut", "tr", "nl", "tac",
        "basename", "dirname", "realpath", "readlink",
        "stat", "file", "du", "df",
        "whoami", "uname", "id", "which", "type",
        "hexdump", "od", "strings", "jq",
        "sha256sum", "shasum", "md5sum", "cksum",
    }
)  # fmt: skip


def is_read_only_command(command: str) -> bool:
    """Return ``True`` only when a ``bash`` command is provably read-only.

    A command qualifies when it contains no output-redirection / command-
    chaining / command-substitution metacharacters and every stage of its
    ``|`` pipeline begins with an allow-listed read-only command. Anything the
    check cannot parse or prove (quoting that defeats the naive split, an
    unknown leading word, a write-capable command) returns ``False`` so the call
    stays sequential — false negatives are safe, false positives are not.
    """
    if not command or not command.strip():
        return False
    if any(token in command for token in _UNSAFE_SHELL_CHARS):
        return False
    for stage in command.split("|"):
        stage = stage.strip()
        if not stage:
            return False
        try:
            words = shlex.split(stage)
        except ValueError:
            return False  # unbalanced quotes — cannot prove safe
        if not words or words[0] not in _READ_ONLY_COMMANDS:
            return False
    return True


def is_parallel_safe(func_name: str, args: dict[str, Any]) -> bool:
    """Return ``True`` when this exact tool call may overlap other read-only calls.

    ``read_file`` and ``fetch_web`` are read-only for any argument set; ``bash``
    qualifies only when its ``command`` passes :func:`is_read_only_command`.
    Every other tool (``write_file``, ``read_tool_artifact``, ``delegate``,
    unknown names) returns ``False``, keeping it in sequential order.
    """
    if func_name in _READ_ONLY_TOOLS:
        return True
    if func_name == "bash":
        return is_read_only_command(args.get("command", ""))
    return False

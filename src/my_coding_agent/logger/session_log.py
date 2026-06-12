"""Session-log capture: tee ``sys.stderr`` to plain and ANSI-colored log files.

Owns the ``_TeeStream`` that fans every stderr write out to the original stream, a
colored log file (ANSI preserved), and a plain log file (ANSI stripped), plus the
``attach_session_log``/``detach_session_log`` helpers that install and remove it.
"""

import os
import re
import sys
from typing import Any, TextIO

# ── TeeStream — fans out one write to stderr + two log files ──────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _TeeStream:
    """Wraps the original stderr and tees every write to two extra files."""

    def __init__(
        self, original: TextIO, plain_file: TextIO, colored_file: TextIO
    ) -> None:
        self._orig = original
        self._plain = plain_file
        self._colored = colored_file

    # ── core file protocol ───────────────────────────────────────────────────
    def write(self, data: str) -> int:
        self._orig.write(data)
        self._colored.write(data)
        self._plain.write(_ANSI_RE.sub("", data))
        return len(data)

    def flush(self) -> None:
        self._orig.flush()
        self._colored.flush()
        self._plain.flush()

    def fileno(self) -> int:
        return self._orig.fileno()

    # ── attributes that print() / logging / colorama check ──────────────────
    def isatty(self) -> bool:
        return self._orig.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._orig, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._orig, "errors", "replace")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


# ── Session file log helpers ───────────────────────────────────────────────────
# Opaque handle returned by attach_session_log and consumed by detach_session_log:
# (original stderr, plain log file, colored log file).
_SessionLogHandle = tuple[TextIO, TextIO, TextIO]


def attach_session_log(path: str | os.PathLike[str]) -> _SessionLogHandle:
    """Replace sys.stderr with a TeeStream writing to plain + colored log files."""
    import pathlib

    plain_path = pathlib.Path(path)
    colored_path = plain_path.with_name("stderr_colored.log")
    plain_path.parent.mkdir(parents=True, exist_ok=True)

    plain_file = open(plain_path, "a", encoding="utf-8")
    colored_file = open(colored_path, "a", encoding="utf-8")

    original = sys.stderr
    sys.stderr = _TeeStream(original, plain_file, colored_file)
    # NOTE: We do NOT lower the ROOT logger level here (that would override the
    # host application's logging — CONTRIBUTE.md §31). The package's own loggers
    # are created at DEBUG by ``get_logger``, so every level still reaches the
    # TeeStream via their handlers.
    return (original, plain_file, colored_file)


def detach_session_log(handle: _SessionLogHandle) -> None:
    """Restore sys.stderr and close the log files."""
    original, plain_file, colored_file = handle
    sys.stderr.flush()
    sys.stderr = original
    plain_file.close()
    colored_file.close()

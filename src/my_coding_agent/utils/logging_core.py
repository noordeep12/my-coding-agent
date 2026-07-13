"""Logging primitives and stderr session-log capture.

Owns the custom level numbers and their registration, the ``_PackageLogger``
subclass that adds ``tool``/``api``/``llm`` convenience methods without mutating
the global ``logging.Logger`` class, the ``ColoredFormatter``, the
``DynamicStderrHandler`` that follows ``sys.stderr`` replacement, the public
``get_logger`` factory, and the ``_TeeStream`` / ``attach_session_log`` /
``detach_session_log`` helpers that fan stderr writes to plain + colored log files.
"""

import logging
import os
import re
import sys
from typing import Any, TextIO

from colorama import Back, Fore, Style  # type: ignore[import-untyped]

# ── Custom level numbers ──────────────────────────────────────────────────────
TOOL = 15
API = 25
LLM = 35

_CUSTOM_LEVELS: dict[str, int] = {
    "TOOL": TOOL,
    "API": API,
    "LLM": LLM,
}

# ── Colors per level (full-line coloring) ─────────────────────────────────────
_LEVEL_COLORS: dict[str, str] = {
    "TOOL": Fore.MAGENTA,
    "API": Fore.CYAN,
    "LLM": Fore.YELLOW,
    "DEBUG": Style.DIM,
    "INFO": Fore.WHITE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Back.WHITE,
}


def _register_level_names() -> None:
    """Map the custom level numbers to names (does not mutate ``logging.Logger``)."""
    for name, number in _CUSTOM_LEVELS.items():
        logging.addLevelName(number, name)


_register_level_names()


class _PackageLogger(logging.Logger):
    """Logger subclass that adds the project's TOOL/API/LLM convenience methods.

    Defining these methods on a subclass (used only for the package's own named
    loggers) avoids monkeypatching the global ``logging.Logger`` class, so a host
    application's loggers are unaffected. See CONTRIBUTE.md §31.
    """

    def tool(self, msg: object, *args: object, **kwargs: Any) -> None:
        if self.isEnabledFor(TOOL):
            self._log(TOOL, msg, args, **kwargs)

    def api(self, msg: object, *args: object, **kwargs: Any) -> None:
        if self.isEnabledFor(API):
            self._log(API, msg, args, **kwargs)

    def llm(self, msg: object, *args: object, **kwargs: Any) -> None:
        if self.isEnabledFor(LLM):
            self._log(LLM, msg, args, **kwargs)


# ── Formatter ─────────────────────────────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    """Format log records with a per-level ANSI color wrapping the whole line."""

    def format(self, record: logging.LogRecord) -> str:
        """Attach the level's color/reset codes to the record and format it."""
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.color = color
        record.reset = Style.RESET_ALL
        return super().format(record)


# ── Dynamic stderr handler (follows sys.stderr after it is replaced) ───────────
class DynamicStderrHandler(logging.StreamHandler):
    """StreamHandler that writes to the current sys.stderr, not a captured ref."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, value: TextIO) -> None:
        pass  # ignore — always use live sys.stderr


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.DEBUG) -> _PackageLogger:
    """Return the package's named logger, carrying the TOOL/API/LLM methods.

    The logger is created as a ``_PackageLogger`` (only for this name) so the
    custom level methods are available without mutating the global
    ``logging.Logger`` class. Only the package's own named logger is configured;
    the root logger is left untouched (CONTRIBUTE.md §31).
    """
    formatter = ColoredFormatter(
        "{color}{asctime} | {levelname} | {message}{reset}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = DynamicStderrHandler()
    handler.setFormatter(formatter)

    logger = _get_package_logger(name)
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def _get_package_logger(name: str) -> _PackageLogger:
    """Fetch ``name`` as a ``_PackageLogger`` without changing the global class.

    ``logging.setLoggerClass`` is swapped only for the duration of the lookup so
    no other logger in the interpreter is affected. If the logger already exists
    under a different class (e.g. created before this call), its custom methods
    are bound on the instance as a fallback.
    """
    previous = logging.getLoggerClass()
    logging.setLoggerClass(_PackageLogger)
    try:
        logger = logging.getLogger(name)
    finally:
        logging.setLoggerClass(previous)
    if not isinstance(logger, _PackageLogger):
        # Pre-existing plain Logger under this name — bind methods on the instance.
        logger.tool = _PackageLogger.tool.__get__(logger)  # type: ignore[attr-defined]
        logger.api = _PackageLogger.api.__get__(logger)  # type: ignore[attr-defined]
        logger.llm = _PackageLogger.llm.__get__(logger)  # type: ignore[attr-defined]
    return logger  # type: ignore[return-value]


# ── Session-log capture (stderr tee) ──────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _TeeStream:
    """Wraps the original stderr and tees every write to two extra files."""

    def __init__(
        self, original: TextIO, plain_file: TextIO, colored_file: TextIO
    ) -> None:
        self._orig = original
        self._plain = plain_file
        self._colored = colored_file

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


# Opaque handle returned by attach_session_log and consumed by detach_session_log.
_SessionLogHandle = tuple[TextIO, TextIO, TextIO]


def attach_session_log(path: str | os.PathLike[str]) -> _SessionLogHandle:
    """Replace sys.stderr with a TeeStream writing to plain + colored log files."""
    import pathlib

    plain_path = pathlib.Path(path)
    colored_path = plain_path.with_name("stderr_colored.log")
    plain_path.parent.mkdir(parents=True, exist_ok=True)

    plain_file = open(plain_path, "a", encoding="utf-8")  # noqa: SIM115 -- handle outlives this function, returned to caller
    try:
        colored_file = open(colored_path, "a", encoding="utf-8")  # noqa: SIM115 -- handle outlives this function, returned to caller
    except Exception:
        plain_file.close()
        raise

    original = sys.stderr
    sys.stderr = _TeeStream(original, plain_file, colored_file)
    return (original, plain_file, colored_file)


def detach_session_log(handle: _SessionLogHandle) -> None:
    """Restore sys.stderr and close the log files."""
    original, plain_file, colored_file = handle
    sys.stderr.flush()
    sys.stderr = original
    plain_file.close()
    colored_file.close()

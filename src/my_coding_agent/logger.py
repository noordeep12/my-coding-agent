"""Logging, stderr tee/session-log capture, and terminal-UI rendering.

Provides the package logger with custom TOOL/API/LLM levels, a stderr handler that
follows ``sys.stderr`` replacement, the ``_TeeStream`` that fans session output to
plain and ANSI-colored log files (``attach_session_log``/``detach_session_log``),
and the rich box-drawn ``print_banner`` and ``print_run_summary`` renderers.
"""

import io
import logging
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Any, TextIO

from colorama import Back, Fore, Style  # type: ignore[import-untyped]
from rich.console import Console
from rich.markdown import Markdown

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


# ── Git helper ────────────────────────────────────────────────────────────────
def _git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ── Startup banner ────────────────────────────────────────────────────────────
def print_banner(  # noqa: C901
    label: str,
    model: str,
    tools: list,
    context_window: int | None = None,
    n_messages: int = 0,
    context_reset_threshold: float = 0.75,
    session_id: str | None = None,
) -> None:
    """Render the startup banner box to stderr.

    Draw the ASCII logo and a panel of run metadata — model, tool count, context
    window, message count, reset threshold, workspace, git branch, session id, and
    timestamp — using box-drawing characters and color.

    Args:
        label: Agent label shown in the banner title.
        model: Model id being run.
        tools: Tool definitions; their count and names are listed.
        context_window: Model context window in tokens, or None if unknown.
        n_messages: Number of seed messages already in the conversation.
        context_reset_threshold: Fraction of the window that triggers a handoff.
        session_id: Session identifier; a random one is shown if omitted.
    """
    W = 68  # visible inner width (between the two ║)
    R: str = Style.RESET_ALL
    BORDER: str = Fore.CYAN + Style.BRIGHT
    LABEL: str = Fore.CYAN + Style.BRIGHT
    VALUE: str = Fore.WHITE + Style.BRIGHT
    LOGO_C: str = Fore.CYAN + Style.BRIGHT
    TITLE_C: str = Fore.GREEN + Style.BRIGHT

    ascii_logo = [
        r"  ██████╗ ██████╗ ██████╗ ███████╗",
        r" ██╔════╝██╔═══██╗██╔══██╗██╔════╝",
        r" ██║     ██║   ██║██║  ██║█████╗  ",
        r" ██║     ██║   ██║██║  ██║██╔══╝  ",
        r" ╚██████╗╚██████╔╝██████╔╝███████╗",
        r"  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
        r"     MY CODING AGENT  v0.1.0       ",
    ]

    branch = _git_branch()
    session_id = session_id or uuid.uuid4().hex[:12]
    workspace = os.getcwd()
    ctx_str = f"{context_window:,}" if context_window else "unknown"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reset_str = f"{context_reset_threshold * 100:.0f}%"

    top = BORDER + "╔" + "═" * W + "╗" + R
    bottom = BORDER + "╚" + "═" * W + "╝" + R
    mid = BORDER + "╠" + "═" * W + "╣" + R

    def empty_row() -> str:
        return BORDER + "║" + " " * W + "║" + R

    def logo_row(text: str) -> str:
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return (
            BORDER
            + "║"
            + " " * pad_l
            + LOGO_C
            + text
            + R
            + " " * pad_r
            + BORDER
            + "║"
            + R
        )

    def title_row(text: str) -> str:
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return (
            BORDER
            + "║"
            + " " * pad_l
            + TITLE_C
            + text
            + R
            + " " * pad_r
            + BORDER
            + "║"
            + R
        )

    def info_row(lbl1: str, val1: str, lbl2: str = "", val2: str = "") -> str:
        HALF = W // 2
        left_vis = f"  {lbl1}: {val1}"
        left_pad = HALF - len(left_vis)
        left_col = f"  {LABEL}{lbl1}{R}: {VALUE}{val1}{R}" + " " * max(left_pad, 1)
        if lbl2:
            right_vis = f"{lbl2}: {val2}  "
            right_pad = HALF - len(right_vis)
            right_col = (
                " " * max(right_pad, 1) + f"{LABEL}{lbl2}{R}: {VALUE}{val2}{R}  "
            )
        else:
            right_col = " " * HALF
        return BORDER + "║" + left_col + right_col + BORDER + "║" + R

    def full_row(lbl: str, val: str) -> str:
        vis = f"  {lbl}: {val}"
        pad = W - len(vis)
        inner = f"  {LABEL}{lbl}{R}: {VALUE}{val}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    def tool_row(t: dict) -> str:
        name = t["function"]["name"]
        params = ", ".join(t["function"]["parameters"]["properties"].keys())
        sig = f"{name}({params})"
        max_sig = W - 6
        if len(sig) > max_sig:
            sig = sig[: max_sig - 1] + "…"
        vis = f"    {sig}"
        pad = W - len(vis)
        inner = f"    {VALUE}{sig}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    lines = ["", top, empty_row()]
    for line in ascii_logo:
        lines.append(logo_row(line))
    lines += [empty_row(), mid, empty_row()]

    # Agent title
    agent_title = f"▸  {label.upper()}"
    lines.append(title_row(agent_title))
    lines += [empty_row(), mid, empty_row()]

    # Metrics
    lines += [
        info_row("MODEL", model[:28], "BRANCH", branch),
        info_row("SESSION", session_id, "TIME", timestamp),
        info_row("CONTEXT", ctx_str, "RESET AT", reset_str),
        info_row("MESSAGES", str(n_messages), "TOOLS", str(len(tools))),
        full_row(
            "WORKSPACE",
            workspace if len(workspace) <= W - 16 else "…" + workspace[-(W - 17) :],
        ),
    ]

    # Tools section
    if tools:
        lines += [empty_row(), mid, empty_row()]
        for t in tools:
            lines.append(tool_row(t))

    lines += [empty_row(), bottom, ""]
    print("\n".join(lines), file=sys.stderr)


# ── Run summary ───────────────────────────────────────────────────────────────
_SUMMARY_W = 110  # box inner width (chart + session header)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _SummaryStyle:
    """Shared box-drawing constants and primitive row builders for the summary.

    Holds what was previously a block of closure locals inside
    ``print_run_summary`` so the row helpers can live at module scope (keeping the
    public function's McCabe complexity within the §38 ceiling). Output is byte
    for byte identical to the former inline version.
    """

    def __init__(self) -> None:
        self.W = _SUMMARY_W
        self.R: str = Style.RESET_ALL
        self.BORDER: str = Fore.CYAN + Style.BRIGHT
        self.LABEL: str = Fore.CYAN + Style.BRIGHT
        self.VALUE: str = Fore.WHITE + Style.BRIGHT
        self.OK: str = Fore.GREEN
        self.FAIL: str = Fore.RED
        self.WARN: str = Fore.YELLOW + Style.BRIGHT
        self.ansi_re = _ANSI
        self.top = self.BORDER + "╔" + "═" * self.W + "╗" + self.R
        self.bottom = self.BORDER + "╚" + "═" * self.W + "╝" + self.R
        self.mid = self.BORDER + "╠" + "═" * self.W + "╣" + self.R

    def empty_row(self) -> str:
        return self.BORDER + "║" + " " * self.W + "║" + self.R

    def title_row(self, text: str) -> str:
        pad_l = (self.W - len(text)) // 2
        pad_r = self.W - pad_l - len(text)
        return (
            self.BORDER
            + "║"
            + " " * pad_l
            + self.VALUE
            + text
            + self.R
            + " " * pad_r
            + self.BORDER
            + "║"
            + self.R
        )

    def metric_row1(self, lbl: str, val: str) -> str:
        """Full-width single metric row."""
        vis = f"  {lbl}: {val}"
        pad = self.W - len(vis)
        inner = f"  {self.LABEL}{lbl}{self.R}: {self.VALUE}{val}{self.R}" + " " * max(
            pad, 0
        )
        return self.BORDER + "║" + inner + self.BORDER + "║" + self.R


def _tool_call_rows(
    s: _SummaryStyle,
    index: int,
    name: str,
    args: dict,
    ok: bool,
    status: str = "",
    artifact: bool = False,
    truncated: bool = False,
) -> list[str]:
    """One or more box rows for a single tool call, wrapping long args."""
    W, R, BORDER, VALUE = s.W, s.R, s.BORDER, s.VALUE
    SKIP = Fore.YELLOW + Style.BRIGHT
    if status == "skipped":
        status_vis = "⊘"
        status_col = SKIP + status_vis + R
    elif ok:
        status_vis = "✓"
        status_col = s.OK + status_vis + R
    else:
        status_vis = "✗"
        status_col = s.FAIL + status_vis + R
    badges = ""
    badges_vis = ""
    if artifact:
        badges_vis += " [artifact]"
        badges += f" {Fore.CYAN}[artifact]{R}"
    if truncated:
        badges_vis += " [truncated]"
        badges += f" {s.WARN}[truncated]{R}"
    # margin: 2 left, 3 right (space + status + space)
    MARGIN = 5
    args_raw = ", ".join(f"{k}={repr(v)}" for k, v in args.items()) if args else ""
    prefix = f"{index:3}. {name}("
    suffix = ")" + badges_vis
    content_w = W - MARGIN - len(prefix) - len(suffix)

    # split args_raw into lines of content_w
    chunks: list[str] = []
    remaining = args_raw
    while len(remaining) > content_w:
        # try to break at a comma+space boundary within the limit
        cut = remaining.rfind(", ", 0, content_w)
        if cut == -1:
            cut = content_w
        else:
            cut += 2  # include the ", "
        chunks.append(remaining[:cut])
        remaining = "    " + remaining[cut:]  # indent continuation
    chunks.append(remaining)

    colored_suffix = ")" + badges
    rows: list[str] = []
    for ci, chunk in enumerate(chunks):
        is_last = ci == len(chunks) - 1
        if ci == 0:
            line_vis = prefix + chunk + (suffix if is_last else "")
            line_col = f"{VALUE}{prefix}{chunk}{R}" + (
                colored_suffix if is_last else ""
            )
        else:
            line_vis = " " * len(prefix) + chunk + (suffix if is_last else "")
            line_col = (
                " " * len(prefix)
                + f"{VALUE}{chunk}{R}"
                + (colored_suffix if is_last else "")
            )
        pad = W - 2 - len(line_vis) - (3 if is_last else 2)
        if is_last:
            inner = f"  {line_col}" + " " * max(pad, 1) + status_col + "  "
        else:
            inner = f"  {line_col}" + " " * max(pad + 3, 0)
        rows.append(BORDER + "║" + inner + BORDER + "║" + R)
    return rows


def _markdown_rows(s: _SummaryStyle, md_text: str) -> list[str]:
    """Render markdown via rich and box each output line."""
    W, R, BORDER = s.W, s.R, s.BORDER
    buf = io.StringIO()
    console = Console(
        file=buf, width=W - 4, force_terminal=True, highlight=False, no_color=False
    )
    console.print(Markdown(md_text))
    rendered = buf.getvalue()
    rows: list[str] = []
    for line in rendered.splitlines():
        visible_len = len(s.ansi_re.sub("", line))
        pad = (W - 4) - visible_len
        rows.append(BORDER + "║  " + line + " " * max(pad, 0) + "  " + BORDER + "║" + R)
    return rows or [s.empty_row()]


def _token_chart_rows(s: _SummaryStyle, llm_calls: list | None) -> list[str]:
    """Render the per-main-call token chart (and harness-event annotation)."""
    W, R, BORDER = s.W, s.R, s.BORDER
    usage = llm_calls or []
    if len(usage) < 1:
        return [s.metric_row1("TOKEN CHART", "no data")]
    import plotext as plt  # type: ignore[import-untyped]

    _KIND_LABEL = {
        "router": "router",
        "summarizer": "summarizer",
        "correction": "correction",
        "handoff": "handoff",
    }

    main_calls = [u for u in usage if u.get("kind", "main") == "main"]
    internal_calls = [u for u in usage if u.get("kind", "main") != "main"]

    if not main_calls:
        return [s.metric_row1("TOKEN CHART", "no main-agent calls recorded")]

    # Sequential index (1-based) among main calls only
    main_x = list(range(1, len(main_calls) + 1))
    prompt_vals = [u["prompt"] for u in main_calls]
    comp_vals = [u["completion"] for u in main_calls]
    all_vals = sorted(set(prompt_vals + comp_vals))

    chart_w = W - 6
    chart_h = 20
    plt.clf()
    plt.plot_size(chart_w, chart_h)
    plt.theme("dark")
    plt.plot(main_x, prompt_vals, label="prompt", color="cyan+", marker="braille")
    plt.plot(main_x, comp_vals, label="completion", color="green+", marker="braille")
    plt.xticks(main_x)
    plt.yticks(all_vals, [f"{v:,}" for v in all_vals])
    plt.xlabel("Main LLM Call #")
    plt.ylabel("Tokens")
    plt.title("Token consumption per main agent call")
    chart_str = plt.build()

    rows: list[str] = []
    for line in chart_str.splitlines():
        if not s.ansi_re.sub("", line).strip():
            continue
        visible = len(s.ansi_re.sub("", line))
        pad = (W - 4) - visible
        rows.append(BORDER + "║  " + line + " " * max(pad, 0) + "  " + BORDER + "║" + R)

    # Annotation row for internal (harness) calls
    if internal_calls:
        # Group by kind, list call numbers
        from collections import defaultdict

        by_kind: dict = defaultdict(list)
        for c in internal_calls:
            by_kind[c.get("kind", "internal")].append(str(c["call"]))
        parts = [
            f"{_KIND_LABEL.get(k, k)}: #{', #'.join(nums)}"
            for k, nums in by_kind.items()
        ]
        note = "harness events — " + "  ·  ".join(parts)
        note_vis = f"  {note}"
        pad = W - len(note_vis)
        ANNOT = Fore.YELLOW + Style.DIM
        rows.append(
            BORDER + "║" + f"  {ANNOT}{note}{R}" + " " * max(pad, 0) + BORDER + "║" + R
        )

    return rows or [s.empty_row()]


def _handoff_rows(
    s: _SummaryStyle, index: int, h: dict, context_window: int | None
) -> list[str]:
    """Render one handoff (context reset) event as box rows."""
    W, R, BORDER, VALUE, WARN = s.W, s.R, s.BORDER, s.VALUE, s.WARN
    rows: list[str] = []

    # Line 1: index, step, context % at trigger
    line1_vis = (
        f"  {index}. step {h['step']} — {h['ctx_pct']:.1f}% ctx used "
        f"({h['ctx_tokens']:,} / {context_window:,} tok)"
    )
    pad1 = W - len(line1_vis)
    rows.append(
        BORDER + "║" + f"  {WARN}{index}.{R} step {VALUE}{h['step']}{R} — "
        f"{WARN}{h['ctx_pct']:.1f}%{R} ctx used "
        f"({VALUE}{h['ctx_tokens']:,}{R} / {VALUE}{context_window:,}{R} tok)"
        + " " * max(pad1, 0)
        + BORDER
        + "║"
        + R
    )

    # Line 2: threshold that triggered it
    line2_vis = f"     trigger: prompt_tokens >= {h['threshold']:.0f}% threshold"
    pad2 = W - len(line2_vis)
    rows.append(
        BORDER
        + "║"
        + f"     trigger: prompt_tokens >= {WARN}{h['threshold']:.0f}%{R} threshold"
        + " " * max(pad2, 0)
        + BORDER
        + "║"
        + R
    )

    # Line 3: saved path (truncated to fit)
    path_str = h.get("path", "")
    max_path = W - 12
    if len(path_str) > max_path:
        path_str = "…" + path_str[-(max_path - 1) :]
    line3_vis = f"     saved  : {path_str}"
    pad3 = W - len(line3_vis)
    rows.append(
        BORDER
        + "║"
        + f"     saved  : {VALUE}{path_str}{R}"
        + " " * max(pad3, 0)
        + BORDER
        + "║"
        + R
    )

    return rows


def _tool_calls_section(s: _SummaryStyle, records: list) -> list[str]:
    """Box rows for the TOOL CALLS detail list (empty if there are no records)."""
    if not records:
        return []
    lines = [s.empty_row()]
    for i, r in enumerate(records, start=1):
        lines.extend(
            _tool_call_rows(
                s,
                i,
                r["name"],
                r["args"],
                r["ok"],
                status=r.get("status", ""),
                artifact=r.get("artifact", False),
                truncated=r.get("truncated", False),
            )
        )
    return lines


def _context_resets_section(
    s: _SummaryStyle, handoffs: list, context_window: int | None
) -> list[str]:
    """Box rows for the CONTEXT RESETS section (one block per handoff)."""
    if not handoffs:
        return [
            s.empty_row(),
            s.mid,
            s.empty_row(),
            s.metric_row1("CONTEXT RESETS", "0  (no reset triggered)"),
        ]
    lines = [
        s.empty_row(),
        s.mid,
        s.empty_row(),
        s.metric_row1("CONTEXT RESETS", str(len(handoffs))),
        s.empty_row(),
    ]
    for i, h in enumerate(handoffs, start=1):
        lines.extend(_handoff_rows(s, i, h, context_window))
        if i < len(handoffs):
            lines.append(s.empty_row())
    return lines


def _tool_count_label(records: list) -> str:
    """Summarize tool outcomes as e.g. ``3 (2 ok, 1 failed)`` or ``0``."""
    if not records:
        return "0"
    n_ok = sum(1 for r in records if r["ok"])
    n_skip = sum(1 for r in records if r.get("status") == "skipped")
    n_fail = len(records) - n_ok - n_skip
    parts = [f"{n_ok} ok"]
    if n_fail:
        parts.append(f"{n_fail} failed")
    if n_skip:
        parts.append(f"{n_skip} skipped")
    return f"{len(records)} ({', '.join(parts)})"


def print_run_summary(
    steps: int,
    max_steps: int,
    stop_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    context_window: int | None = None,
    elapsed_seconds: float = 0.0,
    tool_records: list | None = None,
    handoff_records: list | None = None,
    agent_name: str = "Agent",
    last_message: str = "",
    last_prompt_tokens: int = 0,
    llm_calls: list | None = None,
    model: str = "",
    session_id: str = "",
    started_at: str = "",
    tools: list | None = None,
) -> None:
    """Render the end-of-run summary box to stderr.

    Draw a box reporting the run outcome: step count, stop reason, elapsed time and
    throughput, token totals, a per-call token-consumption chart, the last model
    message, and the tool-call and context-reset sections.

    Args:
        steps: Number of loop steps executed.
        max_steps: Configured step budget.
        stop_reason: Why the loop ended (e.g. ``stop``, ``max_steps``).
        prompt_tokens: Total prompt tokens across the run.
        completion_tokens: Total completion tokens across the run.
        total_tokens: Combined token total.
        context_window: Model context window in tokens, or None if unknown.
        elapsed_seconds: Wall-clock run duration.
        tool_records: Per-tool-call records summarized in the tool section.
        handoff_records: Context-reset/handoff records, if any.
        llm_calls: Per-call usage entries plotted in the token chart.
    """
    s = _SummaryStyle()
    metric_row1 = s.metric_row1

    # ── computed values ────────────────────────────────────────────────────────
    ctx_pct = (
        f" ({last_prompt_tokens / context_window * 100:.1f}% of {context_window:,})"
        if context_window and last_prompt_tokens
        else ""
    )
    mins, secs = divmod(elapsed_seconds, 60)
    elapsed_str = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
    tok_per_sec = (
        f"{completion_tokens / elapsed_seconds:.1f} tok/s"
        if elapsed_seconds > 0
        else "—"
    )
    records = tool_records or []
    tool_count = _tool_count_label(records)
    handoffs = handoff_records or []
    branch = _git_branch()
    workspace = os.getcwd()

    # ── build box ──────────────────────────────────────────────────────────────
    ctx_str = f"{context_window:,}" if context_window else "unknown"
    ws_str = workspace if len(workspace) <= s.W - 16 else "…" + workspace[-(s.W - 17) :]
    tools_str = str(len(tools)) if tools is not None else "—"

    lines: list[str] = [
        "",
        s.top,
        s.empty_row(),
        s.title_row(f"▸  {agent_name.upper()}  —  RUN COMPLETE"),
        s.empty_row(),
        s.mid,
        s.empty_row(),
        # ── session header (mirrors banner) ───────────────────────────────────
        metric_row1("MODEL", model or "—"),
        metric_row1("SESSION", session_id or "—"),
        metric_row1("STARTED", started_at or "—"),
        metric_row1("BRANCH", branch),
        metric_row1("WORKSPACE", ws_str),
        metric_row1("CONTEXT", ctx_str),
        metric_row1("TOOLS", tools_str),
        s.empty_row(),
        s.mid,
        s.empty_row(),
        # ── run metrics ───────────────────────────────────────────────────────
        metric_row1("STEPS", f"{steps} / {max_steps}"),
        metric_row1("STOP REASON", stop_reason),
        metric_row1("ELAPSED", elapsed_str),
        metric_row1("THROUGHPUT", tok_per_sec),
        s.empty_row(),
        s.mid,
        s.empty_row(),
        metric_row1("CONTEXT (last call)", f"{last_prompt_tokens:,} tok{ctx_pct}"),
        metric_row1("INPUT (all calls)", f"{prompt_tokens:,} tok cumulative"),
        metric_row1("COMPLETION (all calls)", f"{completion_tokens:,} tok"),
        metric_row1("TOTAL BILLED", f"{total_tokens:,} tok"),
        s.mid,
        # ── token chart ───────────────────────────────────────────────────────
        *_token_chart_rows(s, llm_calls),
        s.mid,
        metric_row1("TOOL CALLS", tool_count),
    ]

    lines += _tool_calls_section(s, records)
    lines += _context_resets_section(s, handoffs, context_window)

    if last_message:
        lines += [s.empty_row(), s.mid, s.empty_row()]
        lines += _markdown_rows(s, last_message)

    lines += [s.empty_row(), s.bottom, ""]
    print("\n".join(lines), file=sys.stderr)

import io
import os
import re
import sys
import uuid
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

from colorama import Fore, Back, Style
from rich.console import Console
from rich.markdown import Markdown


# ── Custom level numbers ──────────────────────────────────────────────────────
TOOL = 15
API  = 25
LLM  = 35

_CUSTOM_LEVELS: Dict[str, int] = {
    "TOOL": TOOL,
    "API":  API,
    "LLM":  LLM,
}

# ── Colors per level (full-line coloring) ─────────────────────────────────────
_LEVEL_COLORS: Dict[str, str] = {
    "TOOL":     Fore.MAGENTA,
    "API":      Fore.WHITE,
    "LLM":      Fore.YELLOW,
    "DEBUG":    Style.DIM,
    "INFO":     Fore.WHITE,
    "WARNING":  Fore.YELLOW,
    "ERROR":    Fore.RED,
    "CRITICAL": Fore.RED + Back.WHITE,
}


def _register_custom_levels() -> None:
    for name, number in _CUSTOM_LEVELS.items():
        logging.addLevelName(number, name)

        def _make_method(lvl: int):
            def method(self, msg, *args, **kwargs):
                if self.isEnabledFor(lvl):
                    self._log(lvl, msg, args, **kwargs)
            return method

        setattr(logging.Logger, name.lower(), _make_method(number))


_register_custom_levels()


# ── Formatter ─────────────────────────────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    def format(self, record) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.color = color
        record.reset = Style.RESET_ALL
        return super().format(record)


# ── Dynamic stderr handler (follows sys.stderr after it is replaced) ───────────
class DynamicStderrHandler(logging.StreamHandler):
    """StreamHandler that always writes to the current sys.stderr, not a captured reference."""

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass  # ignore — always use live sys.stderr


# ── TeeStream — fans out one write to stderr + two log files ──────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _TeeStream:
    """Wraps the original stderr and tees every write to two extra files."""

    def __init__(self, original, plain_file, colored_file):
        self._orig    = original
        self._plain   = plain_file
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

    def fileno(self):
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

    def __getattr__(self, name):
        return getattr(self._orig, name)


# ── Session file log helpers ───────────────────────────────────────────────────
def attach_session_log(path) -> "_SessionLogHandle":
    """Replace sys.stderr with a TeeStream that also writes to plain + colored log files."""
    import pathlib
    plain_path   = pathlib.Path(path)
    colored_path = plain_path.with_name("stderr_colored.log")
    plain_path.parent.mkdir(parents=True, exist_ok=True)

    plain_file   = open(plain_path,   "a", encoding="utf-8")
    colored_file = open(colored_path, "a", encoding="utf-8")

    original  = sys.stderr
    sys.stderr = _TeeStream(original, plain_file, colored_file)
    # Root logger defaults to WARNING — lower it so all levels reach the TeeStream.
    logging.getLogger().setLevel(logging.DEBUG)
    return (original, plain_file, colored_file)


def detach_session_log(handle) -> None:
    """Restore sys.stderr and close the log files."""
    original, plain_file, colored_file = handle
    sys.stderr.flush()
    sys.stderr = original
    plain_file.close()
    colored_file.close()


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    formatter = ColoredFormatter(
        "{color}{asctime} | {levelname} | {message}{reset}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = DynamicStderrHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Git helper ────────────────────────────────────────────────────────────────
def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ── Startup banner ────────────────────────────────────────────────────────────
def print_banner(
    label: str,
    model: str,
    tools: list,
    context_window: Optional[int] = None,
    n_messages: int = 0,
    context_reset_threshold: float = 0.75,
    session_id: Optional[str] = None,
) -> None:
    W = 68  # visible inner width (between the two ║)
    R = Style.RESET_ALL
    BORDER = Fore.CYAN + Style.BRIGHT
    LABEL  = Fore.CYAN + Style.BRIGHT
    VALUE  = Fore.WHITE + Style.BRIGHT
    LOGO_C = Fore.CYAN + Style.BRIGHT
    TITLE_C = Fore.GREEN + Style.BRIGHT

    ascii_logo = [
        r"  ██████╗ ██████╗ ██████╗ ███████╗",
        r" ██╔════╝██╔═══██╗██╔══██╗██╔════╝",
        r" ██║     ██║   ██║██║  ██║█████╗  ",
        r" ██║     ██║   ██║██║  ██║██╔══╝  ",
        r" ╚██████╗╚██████╔╝██████╔╝███████╗",
        r"  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
        r"     MY CODING AGENT  v0.1.0       ",
    ]

    branch     = _git_branch()
    session_id = session_id or uuid.uuid4().hex[:12]
    workspace  = os.getcwd()
    ctx_str    = f"{context_window:,}" if context_window else "unknown"
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reset_str  = f"{context_reset_threshold * 100:.0f}%"

    top    = BORDER + "╔" + "═" * W + "╗" + R
    bottom = BORDER + "╚" + "═" * W + "╝" + R
    mid    = BORDER + "╠" + "═" * W + "╣" + R

    def empty_row() -> str:
        return BORDER + "║" + " " * W + "║" + R

    def logo_row(text: str) -> str:
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return BORDER + "║" + " " * pad_l + LOGO_C + text + R + " " * pad_r + BORDER + "║" + R

    def title_row(text: str) -> str:
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return BORDER + "║" + " " * pad_l + TITLE_C + text + R + " " * pad_r + BORDER + "║" + R

    def info_row(lbl1: str, val1: str, lbl2: str = "", val2: str = "") -> str:
        HALF = W // 2
        left_vis  = f"  {lbl1}: {val1}"
        left_pad  = HALF - len(left_vis)
        left_col  = f"  {LABEL}{lbl1}{R}: {VALUE}{val1}{R}" + " " * max(left_pad, 1)
        if lbl2:
            right_vis = f"{lbl2}: {val2}  "
            right_pad = HALF - len(right_vis)
            right_col = " " * max(right_pad, 1) + f"{LABEL}{lbl2}{R}: {VALUE}{val2}{R}  "
        else:
            right_col = " " * HALF
        return BORDER + "║" + left_col + right_col + BORDER + "║" + R

    def full_row(lbl: str, val: str) -> str:
        vis   = f"  {lbl}: {val}"
        pad   = W - len(vis)
        inner = f"  {LABEL}{lbl}{R}: {VALUE}{val}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    def tool_row(t: dict) -> str:
        name   = t["function"]["name"]
        params = ", ".join(t["function"]["parameters"]["properties"].keys())
        sig    = f"{name}({params})"
        max_sig = W - 6
        if len(sig) > max_sig:
            sig = sig[:max_sig - 1] + "…"
        vis   = f"    {sig}"
        pad   = W - len(vis)
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
        info_row("MODEL",    model[:28],       "BRANCH",   branch),
        info_row("SESSION",  session_id,        "TIME",     timestamp),
        info_row("CONTEXT",  ctx_str,           "RESET AT", reset_str),
        info_row("MESSAGES", str(n_messages),   "TOOLS",    str(len(tools))),
        full_row("WORKSPACE", workspace if len(workspace) <= W - 16 else "…" + workspace[-(W - 17):]),
    ]

    # Tools section
    if tools:
        lines += [empty_row(), mid, empty_row()]
        for t in tools:
            lines.append(tool_row(t))

    lines += [empty_row(), bottom, ""]
    print("\n".join(lines), file=sys.stderr)


# ── Run summary ───────────────────────────────────────────────────────────────
def print_run_summary(
    steps: int,
    max_steps: int,
    stop_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    context_window: Optional[int] = None,
    elapsed_seconds: float = 0.0,
    tool_records: Optional[list] = None,
    handoff_records: Optional[list] = None,
    agent_name: str = "Agent",
    last_message: str = "",
) -> None:
    W = 68  # same inner width as the startup banner
    R      = Style.RESET_ALL
    BORDER = Fore.CYAN + Style.BRIGHT
    LABEL  = Fore.CYAN + Style.BRIGHT
    VALUE  = Fore.WHITE + Style.BRIGHT
    OK     = Fore.GREEN
    FAIL   = Fore.RED

    top    = BORDER + "╔" + "═" * W + "╗" + R
    bottom = BORDER + "╚" + "═" * W + "╝" + R
    mid    = BORDER + "╠" + "═" * W + "╣" + R

    def empty_row() -> str:
        return BORDER + "║" + " " * W + "║" + R

    def title_row(text: str) -> str:
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return BORDER + "║" + " " * pad_l + VALUE + text + R + " " * pad_r + BORDER + "║" + R

    def _col(lbl: str, val: str, width: int, align: str = "left") -> tuple[str, str]:
        """Return (visible_str, colored_str) both exactly `width` chars wide."""
        vis = f"  {lbl}: {val}"
        if align == "right":
            vis = f"{lbl}: {val}  "
        pad = width - len(vis)
        if align == "right":
            colored = " " * max(pad, 1) + f"{LABEL}{lbl}{R}: {VALUE}{val}{R}  "
            vis_out = " " * max(pad, 1) + vis
        else:
            colored = f"  {LABEL}{lbl}{R}: {VALUE}{val}{R}" + " " * max(pad, 1)
            vis_out = vis + " " * max(pad, 1)
        return vis_out, colored

    def metric_row2(lbl1: str, val1: str, lbl2: str, val2: str) -> str:
        """Two-column metric row — each column gets half the width."""
        H = W // 2
        _, left  = _col(lbl1, val1, H, "left")
        _, right = _col(lbl2, val2, H, "right")
        return BORDER + "║" + left + right + BORDER + "║" + R

    def metric_row1(lbl: str, val: str) -> str:
        """Full-width single metric row."""
        vis = f"  {lbl}: {val}"
        pad = W - len(vis)
        inner = f"  {LABEL}{lbl}{R}: {VALUE}{val}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    def tool_call_rows(index: int, name: str, args: dict, ok: bool, status: str = "", artifact: bool = False, truncated: bool = False) -> List[str]:
        """One or more box rows for a single tool call, wrapping long args."""
        SKIP = Fore.YELLOW + Style.BRIGHT
        if status == "skipped":
            status_vis = "⊘"
            status_col = SKIP + status_vis + R
        elif ok:
            status_vis = "✓"
            status_col = OK + status_vis + R
        else:
            status_vis = "✗"
            status_col = FAIL + status_vis + R
        badges = ""
        badges_vis = ""
        if artifact:
            badges_vis += " [artifact]"
            badges += f" {Fore.CYAN}[artifact]{R}"
        if truncated:
            badges_vis += " [truncated]"
            badges += f" {WARN}[truncated]{R}"
        # margin: 2 left, 3 right (space + status + space)
        MARGIN = 5
        args_raw = ", ".join(f"{k}={repr(v)}" for k, v in args.items()) if args else ""
        prefix = f"{index:3}. {name}("
        suffix = ")" + badges_vis
        content_w = W - MARGIN - len(prefix) - len(suffix)

        # split args_raw into lines of content_w
        chunks: List[str] = []
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
        rows: List[str] = []
        for ci, chunk in enumerate(chunks):
            is_last = ci == len(chunks) - 1
            if ci == 0:
                line_vis = prefix + chunk + (suffix if is_last else "")
                line_col = f"{VALUE}{prefix}{chunk}{R}" + (colored_suffix if is_last else "")
            else:
                line_vis = " " * len(prefix) + chunk + (suffix if is_last else "")
                line_col = " " * len(prefix) + f"{VALUE}{chunk}{R}" + (colored_suffix if is_last else "")
            pad = W - 2 - len(line_vis) - (3 if is_last else 2)
            if is_last:
                inner = f"  {line_col}" + " " * max(pad, 1) + status_col + "  "
            else:
                inner = f"  {line_col}" + " " * max(pad + 3, 0)
            rows.append(BORDER + "║" + inner + BORDER + "║" + R)
        return rows

    def markdown_rows(md_text: str) -> List[str]:
        """Render markdown via rich and box each output line."""
        buf = io.StringIO()
        console = Console(file=buf, width=W - 4, force_terminal=True, highlight=False, no_color=False)
        console.print(Markdown(md_text))
        rendered = buf.getvalue()
        rows: List[str] = []
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        for line in rendered.splitlines():
            visible_len = len(ansi_escape.sub("", line))
            pad = (W - 4) - visible_len
            rows.append(BORDER + "║  " + line + " " * max(pad, 0) + "  " + BORDER + "║" + R)
        return rows or [empty_row()]

    # ── computed values ────────────────────────────────────────────────────────
    ctx_pct     = f" ({prompt_tokens / context_window * 100:.1f}% of {context_window:,})" if context_window and prompt_tokens else ""
    mins, secs  = divmod(elapsed_seconds, 60)
    elapsed_str = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
    tok_per_sec = f"{completion_tokens / elapsed_seconds:.1f} tok/s" if elapsed_seconds > 0 else "—"
    records     = tool_records or []
    n_ok        = sum(1 for r in records if r["ok"])
    n_skip      = sum(1 for r in records if r.get("status") == "skipped")
    n_fail      = len(records) - n_ok - n_skip
    parts       = [f"{n_ok} ok"]
    if n_fail:
        parts.append(f"{n_fail} failed")
    if n_skip:
        parts.append(f"{n_skip} skipped")
    tool_count  = f"{len(records)} ({', '.join(parts)})" if records else "0"
    handoffs    = handoff_records or []

    WARN = Fore.YELLOW + Style.BRIGHT

    def handoff_rows(index: int, h: dict) -> List[str]:
        """Render one handoff event as box rows."""
        rows: List[str] = []

        # Line 1: index, step, context % at trigger
        line1_vis = f"  {index}. step {h['step']} — {h['ctx_pct']:.1f}% ctx used ({h['ctx_tokens']:,} / {context_window:,} tok)"
        pad1 = W - len(line1_vis)
        rows.append(
            BORDER + "║" +
            f"  {WARN}{index}.{R} step {VALUE}{h['step']}{R} — "
            f"{WARN}{h['ctx_pct']:.1f}%{R} ctx used "
            f"({VALUE}{h['ctx_tokens']:,}{R} / {VALUE}{context_window:,}{R} tok)" +
            " " * max(pad1, 0) +
            BORDER + "║" + R
        )

        # Line 2: threshold that triggered it
        line2_vis = f"     trigger: prompt_tokens >= {h['threshold']:.0f}% threshold"
        pad2 = W - len(line2_vis)
        rows.append(
            BORDER + "║" +
            f"     trigger: prompt_tokens >= {WARN}{h['threshold']:.0f}%{R} threshold" +
            " " * max(pad2, 0) +
            BORDER + "║" + R
        )

        # Line 3: saved path (truncated to fit)
        path_str = h.get("path", "")
        max_path = W - 12
        if len(path_str) > max_path:
            path_str = "…" + path_str[-(max_path - 1):]
        line3_vis = f"     saved  : {path_str}"
        pad3 = W - len(line3_vis)
        rows.append(
            BORDER + "║" +
            f"     saved  : {VALUE}{path_str}{R}" +
            " " * max(pad3, 0) +
            BORDER + "║" + R
        )

        return rows

    # ── build box ──────────────────────────────────────────────────────────────
    lines: List[str] = [
        "", top, empty_row(),
        title_row(f"▸  {agent_name.upper()}  —  RUN COMPLETE"),
        empty_row(), mid, empty_row(),
        metric_row1("STEPS",       f"{steps} / {max_steps}"),
        metric_row1("STOP REASON", stop_reason),
        metric_row1("ELAPSED",     elapsed_str),
        metric_row1("THROUGHPUT",  tok_per_sec),
        empty_row(), mid, empty_row(),
        metric_row1("PROMPT",     f"{prompt_tokens:,} tok{ctx_pct}"),
        metric_row1("COMPLETION", f"{completion_tokens:,} tok"),
        metric_row1("TOTAL",      f"{total_tokens:,} tok"),
        empty_row(), mid, empty_row(),
        metric_row1("TOOL CALLS", tool_count),
    ]

    if records:
        lines.append(empty_row())
        for i, r in enumerate(records, start=1):
            lines.extend(tool_call_rows(
                i, r["name"], r["args"], r["ok"],
                status=r.get("status", ""),
                artifact=r.get("artifact", False),
                truncated=r.get("truncated", False),
            ))

    # ── context resets section ─────────────────────────────────────────────────
    if handoffs:
        lines += [empty_row(), mid, empty_row()]
        lines.append(metric_row1("CONTEXT RESETS", str(len(handoffs))))
        lines.append(empty_row())
        for i, h in enumerate(handoffs, start=1):
            lines.extend(handoff_rows(i, h))
            if i < len(handoffs):
                lines.append(empty_row())
    else:
        lines += [empty_row(), mid, empty_row()]
        lines.append(metric_row1("CONTEXT RESETS", "0  (no reset triggered)"))

    if last_message:
        lines += [empty_row(), mid, empty_row()]
        lines += markdown_rows(last_message)

    lines += [empty_row(), bottom, ""]
    print("\n".join(lines), file=sys.stderr)

import os
import sys
import uuid
import logging
import subprocess
from datetime import datetime
from typing import Dict, Optional

from colorama import Fore, Back, Style


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


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    formatter = ColoredFormatter(
        "{color}{asctime} | {levelname} | {message}{reset}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stderr)
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
def print_banner(model: str, tools: list, context_window: Optional[int] = None) -> None:
    W = 68  # visible inner width (between the two ║)
    R = Style.RESET_ALL
    BORDER = Fore.CYAN + Style.BRIGHT
    LABEL  = Fore.CYAN + Style.BRIGHT
    VALUE  = Fore.WHITE + Style.BRIGHT
    LOGO_C = Fore.CYAN + Style.BRIGHT

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
    session_id = uuid.uuid4().hex[:12]
    workspace  = os.getcwd()
    ctx_str    = f"{context_window:,}" if context_window else "unknown"
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_tools    = str(len(tools))

    top   = BORDER + "╔" + "═" * W + "╗" + R
    bottom = BORDER + "╚" + "═" * W + "╝" + R
    mid   = BORDER + "╠" + "═" * W + "╣" + R

    def empty_row() -> str:
        return BORDER + "║" + " " * W + "║" + R

    def logo_row(text: str) -> str:
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return BORDER + "║" + " " * pad_l + LOGO_C + text + R + " " * pad_r + BORDER + "║" + R

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

    def workspace_row(path: str) -> str:
        truncated = path if len(path) <= W - 16 else "…" + path[-(W - 17):]
        vis   = f"  WORKSPACE: {truncated}"
        pad   = W - len(vis)
        inner = f"  {LABEL}WORKSPACE{R}: {VALUE}{truncated}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    lines = ["", top, empty_row()]
    for line in ascii_logo:
        lines.append(logo_row(line))
    lines += [
        empty_row(), mid, empty_row(),
        info_row("MODEL",   model[:28],  "BRANCH",  branch),
        info_row("TOOLS",   n_tools,     "SESSION", session_id),
        info_row("CONTEXT", ctx_str,     "TIME",    timestamp),
        workspace_row(workspace),
        empty_row(), bottom, "",
    ]
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
    agent_name: str = "Agent",
    last_message: str = "",
) -> None:
    W = 68
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
        # pads purely with spaces — no ANSI in the visible text itself
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        return BORDER + "║" + " " * pad_l + VALUE + text + R + " " * pad_r + BORDER + "║" + R

    def metric_row(lbl: str, val: str, lbl2: str = "", val2: str = "") -> str:
        # compute visible widths first, then add color codes — never mix the two
        if lbl2:
            HALF = W // 2
            lv = f"  {lbl}: {val}"
            rv = f"{lbl2}: {val2}  "
            lpad = max(HALF - len(lv), 1)
            rpad = max(HALF - len(rv), 1)
            left  = f"  {LABEL}{lbl}{R}: {VALUE}{val}{R}" + " " * lpad
            right = " " * rpad + f"{LABEL}{lbl2}{R}: {VALUE}{val2}{R}  "
            # clamp total to W
            inner = left + right
            # recalc actual visible len and re-pad if needed
            vis = len(lv) + lpad + rpad + len(rv)
            if vis < W:
                inner += " " * (W - vis)
        else:
            vis = f"  {lbl}: {val}"
            pad = W - len(vis)
            inner = f"  {LABEL}{lbl}{R}: {VALUE}{val}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    def tool_call_row(index: int, name: str, args: dict, ok: bool) -> str:
        # visible layout: "  NN. name(args…)          ✓  "
        # fixed costs: 2 (left margin) + 1 (status) + 2 (right margin) = 5
        status_vis  = "✓" if ok else "✗"
        status_col  = (OK if ok else FAIL) + status_vis + R
        args_raw    = ", ".join(f"{k}={repr(v)[:24]}" for k, v in args.items()) if args else ""
        prefix_vis  = f"{index:2}. {name}("
        suffix_vis  = ")"
        max_args    = W - 5 - len(prefix_vis) - len(suffix_vis) - 1  # -1 for min 1 space pad
        if len(args_raw) > max_args:
            args_vis = args_raw[:max_args - 1] + "…"
        else:
            args_vis = args_raw
        call_vis = prefix_vis + args_vis + suffix_vis
        pad      = W - 2 - len(call_vis) - 1 - 2  # left_margin + call + status + right_margin
        inner = (
            f"  {VALUE}{index:2}. {name}{R}"
            f"({LABEL}{args_vis}{R})"
            + " " * max(pad, 1)
            + status_col
            + "  "
        )
        return BORDER + "║" + inner + BORDER + "║" + R

    def wrap_text_rows(text: str) -> list:
        # wrap long text into W-4 wide lines (2-space indent each side)
        max_w = W - 4
        words, lines_out, line = text.split(), [], ""
        for word in words:
            if len(line) + len(word) + (1 if line else 0) <= max_w:
                line = (line + " " + word).lstrip()
            else:
                if line:
                    lines_out.append(line)
                line = word
        if line:
            lines_out.append(line)
        rows = []
        for l in lines_out:
            pad = W - 2 - len(l)
            rows.append(BORDER + "║" + f"  {VALUE}{l}{R}" + " " * max(pad, 0) + BORDER + "║" + R)
        return rows or [empty_row()]

    # ── compute values ─────────────────────────────────────────────────────────
    ctx_pct = f" ({total_tokens / context_window * 100:.1f}% of {context_window:,})" if context_window else ""
    mins, secs   = divmod(elapsed_seconds, 60)
    elapsed_str  = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
    tok_per_sec  = f"{completion_tokens / elapsed_seconds:.1f} tok/s" if elapsed_seconds > 0 else "—"
    records      = tool_records or []
    n_ok         = sum(1 for r in records if r["ok"])
    n_fail       = len(records) - n_ok
    tool_count   = f"{len(records)} ({n_ok} ok, {n_fail} failed)" if records else "0"

    # ── build box ──────────────────────────────────────────────────────────────
    lines: list = [
        "", top, empty_row(),
        title_row(f"▸  {agent_name.upper()}  —  RUN COMPLETE"),
        empty_row(), mid, empty_row(),
        metric_row("STEPS",      f"{steps} / {max_steps}",  "STOP REASON", stop_reason),
        metric_row("ELAPSED",    elapsed_str,                "THROUGHPUT",  tok_per_sec),
        empty_row(), mid, empty_row(),
        metric_row("PROMPT",     f"{prompt_tokens:,} tok",   "COMPLETION",  f"{completion_tokens:,} tok"),
        metric_row("TOTAL",      f"{total_tokens:,} tok{ctx_pct}"),
        empty_row(), mid, empty_row(),
        metric_row("TOOL CALLS", tool_count),
    ]

    if records:
        lines.append(empty_row())
        for i, r in enumerate(records, start=1):
            lines.append(tool_call_row(i, r["name"], r["args"], r["ok"]))

    if last_message:
        lines += [empty_row(), mid, empty_row()]
        lines += wrap_text_rows(last_message)

    lines += [empty_row(), bottom, ""]
    print("\n".join(lines), file=sys.stderr)

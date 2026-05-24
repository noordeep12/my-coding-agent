import os
import sys
import uuid
import logging
import subprocess
from datetime import datetime
from typing import Dict, Optional

from colorama import Fore, Back, Style


# ── Custom level numbers ──────────────────────────────────────────────────────
TRACE        = 5
MEMORY_READ  = 12
TOOL_CALL    = 15
MEMORY_WRITE = 18
MEMORY_TRIM  = 22
API_CALL     = 25
AGENT_STEP   = 35
RETRY        = 45
LLM_REASONING = 28
LLM_OUTPUT    = 32
LLM_PARSE     = 38

_CUSTOM_LEVELS: Dict[str, int] = {
    "TRACE":         TRACE,
    "MEMORY_READ":   MEMORY_READ,
    "TOOL_CALL":     TOOL_CALL,
    "MEMORY_WRITE":  MEMORY_WRITE,
    "MEMORY_TRIM":   MEMORY_TRIM,
    "API_CALL":      API_CALL,
    "LLM_REASONING": LLM_REASONING,
    "LLM_OUTPUT":    LLM_OUTPUT,
    "AGENT_STEP":    AGENT_STEP,
    "LLM_PARSE":     LLM_PARSE,
    "RETRY":         RETRY,
}

# ── Colors per level ──────────────────────────────────────────────────────────
_LEVEL_COLORS: Dict[str, str] = {
    "TRACE":         Fore.WHITE,
    "MEMORY_READ":   Fore.BLUE,
    "TOOL_CALL":     Fore.MAGENTA + Style.BRIGHT,
    "MEMORY_WRITE":  Fore.BLUE + Style.BRIGHT,
    "MEMORY_TRIM":   Fore.YELLOW,
    "API_CALL":      Fore.CYAN + Style.BRIGHT,
    "LLM_REASONING": Fore.YELLOW + Style.BRIGHT,
    "LLM_OUTPUT":    Fore.CYAN,
    "AGENT_STEP":    Fore.GREEN + Style.BRIGHT,
    "LLM_PARSE":     Fore.MAGENTA,
    "RETRY":         Fore.RED + Style.BRIGHT,
    "DEBUG":        Fore.CYAN,
    "INFO":         Fore.GREEN,
    "WARNING":      Fore.YELLOW,
    "ERROR":        Fore.RED,
    "CRITICAL":     Fore.RED + Back.WHITE + Style.BRIGHT,
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
        record.color  = _LEVEL_COLORS.get(record.levelname, "")
        record.reset  = Style.RESET_ALL
        return super().format(record)


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    formatter = ColoredFormatter(
        "{asctime} {color}| {levelname:<12}{reset} | {name} | {message}",
        style="{",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
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
    print("\n".join(lines))

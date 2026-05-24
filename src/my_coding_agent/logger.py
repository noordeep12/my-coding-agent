import os
import sys
import uuid
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict

from colorama import Fore, Back, Style


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.colors = colors if colors else {}

    def format(self, record) -> str:
        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL
        return super().format(record)


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


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

    def border_line(char="═") -> str:
        return BORDER + "╔" + char * W + "╗" + R if char == "═" else \
               BORDER + "╠" + char * W + "╣" + R

    top    = BORDER + "╔" + "═" * W + "╗" + R
    bottom = BORDER + "╚" + "═" * W + "╝" + R
    mid    = BORDER + "╠" + "═" * W + "╣" + R

    def bordered(visible_content: str, visible_len: int) -> str:
        pad = W - visible_len
        return BORDER + "║" + visible_content + " " * max(pad, 0) + BORDER + "║" + R

    def empty_row() -> str:
        return BORDER + "║" + " " * W + "║" + R

    def logo_row(text: str) -> str:
        # text is plain ascii art — center it
        pad_l = (W - len(text)) // 2
        pad_r = W - pad_l - len(text)
        content = " " * pad_l + LOGO_C + text + R
        return BORDER + "║" + content + " " * pad_r + BORDER + "║" + R

    def info_row(lbl1: str, val1: str, lbl2: str = "", val2: str = "") -> str:
        # Each half is 34 chars of visible space; 2-char left margin, 2-char right margin
        HALF = W // 2  # 34
        left_vis  = f"  {lbl1}: {val1}"
        left_pad  = HALF - len(left_vis)
        left_colored = f"  {LABEL}{lbl1}{R}: {VALUE}{val1}{R}" + " " * max(left_pad, 1)

        if lbl2:
            right_vis = f"{lbl2}: {val2}  "
            right_pad = HALF - len(right_vis)
            right_colored = " " * max(right_pad, 1) + f"{LABEL}{lbl2}{R}: {VALUE}{val2}{R}  "
        else:
            right_colored = " " * (HALF)

        return BORDER + "║" + left_colored + right_colored + BORDER + "║" + R

    def workspace_row(path: str) -> str:
        truncated = path if len(path) <= W - 16 else "…" + path[-(W - 17):]
        vis   = f"  WORKSPACE: {truncated}"
        pad   = W - len(vis)
        inner = f"  {LABEL}WORKSPACE{R}: {VALUE}{truncated}{R}" + " " * max(pad, 0)
        return BORDER + "║" + inner + BORDER + "║" + R

    lines = [
        "",
        top,
        empty_row(),
    ]
    for line in ascii_logo:
        lines.append(logo_row(line))
    lines += [
        empty_row(),
        mid,
        empty_row(),
        info_row("MODEL",   model[:28],  "BRANCH",  branch),
        info_row("TOOLS",   n_tools,     "SESSION", session_id),
        info_row("CONTEXT", ctx_str,     "TIME",    timestamp),
        workspace_row(workspace),
        empty_row(),
        bottom,
        "",
    ]

    print("\n".join(lines))


def get_logger(name: str) -> logging.Logger:
    """Return a logger with colored output attached."""
    formatter = ColoredFormatter(
        "{asctime} |{color} {levelname:10} {reset}| {name} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        colors={
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
        },
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

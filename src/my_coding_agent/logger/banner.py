"""Startup-banner renderer and the shared git-branch helper.

Owns the rich box-drawn ``print_banner`` renderer and its row-closure helpers,
plus the ``_git_branch`` subprocess helper shared with the run-summary renderer.
Pure presentation — it reads no logging state and only writes formatted text to
``sys.stderr``.
"""

import os
import subprocess
import sys
import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from colorama import Fore, Style  # type: ignore[import-untyped]

# Resolve the package version from installed metadata (single source of truth:
# pyproject [project] version) rather than reimporting the top-level package,
# which would create a circular import — banner is loaded while
# ``my_coding_agent.__init__`` is still executing.
try:
    __version__ = version("my-coding-agent")
except PackageNotFoundError:  # not installed (e.g. source tree without install)
    __version__ = "0.0.0"


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
        f"MY CODING AGENT  v{__version__}",
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
    # Intentional terminal-UI output: render the ANSI box directly to stderr,
    # deliberately bypassing the logger so the box is not prefixed by the
    # ColoredFormatter's "timestamp | LEVEL |" header (CONTRIBUTE.md §31/§36).
    sys.stderr.write("\n".join(lines) + "\n")

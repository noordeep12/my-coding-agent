"""Terminal UI renderers: startup banner and end-of-run summary.

Both renderers share the same box-drawing design language and write directly to
``sys.stderr``, bypassing the logger so the ANSI boxes are not prefixed by the
``ColoredFormatter`` timestamp/level header (CONTRIBUTE.md §31/§36). The
``_git_branch`` helper is shared between the two renderers and lives here.
"""

import io
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from colorama import Fore, Style  # type: ignore[import-untyped]
from rich.console import Console
from rich.markdown import Markdown

from .logging_core import _ANSI_RE as _ANSI

try:
    __version__ = version("my-coding-agent")
except PackageNotFoundError:
    __version__ = "0.0.0"


# ── Shared helper ─────────────────────────────────────────────────────────────


def _git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607 -- relies on `git` from PATH, standard dev tooling
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
    W = 68
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

    agent_title = f"▸  {label.upper()}"
    lines.append(title_row(agent_title))
    lines += [empty_row(), mid, empty_row()]

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

    if tools:
        lines += [empty_row(), mid, empty_row()]
        for t in tools:
            lines.append(tool_row(t))

    lines += [empty_row(), bottom, ""]
    # Intentional terminal-UI output: render the ANSI box directly to stderr,
    # deliberately bypassing the logger so the box is not prefixed by the
    # ColoredFormatter's "timestamp | LEVEL |" header (CONTRIBUTE.md §31/§36).
    sys.stderr.write("\n".join(lines) + "\n")


# ── End-of-run summary ────────────────────────────────────────────────────────

_SUMMARY_W = 110


class _SummaryStyle:
    """Box-drawing constants and primitive row builders for the summary."""

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
    MARGIN = 5
    args_raw = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
    prefix = f"{index:3}. {name}("
    suffix = ")" + badges_vis
    content_w = W - MARGIN - len(prefix) - len(suffix)

    chunks: list[str] = []
    remaining = args_raw
    while len(remaining) > content_w:
        cut = remaining.rfind(", ", 0, content_w)
        if cut == -1:
            cut = content_w
        else:
            cut += 2
        chunks.append(remaining[:cut])
        remaining = "    " + remaining[cut:]
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

    if internal_calls:
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
    """Render one handoff event as box rows."""
    W, R, BORDER, VALUE, WARN = s.W, s.R, s.BORDER, s.VALUE, s.WARN
    rows: list[str] = []

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
    """Box rows for the TOOL CALLS detail list."""
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
    """Box rows for the CONTEXT RESETS section."""
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


def _resource_rollup_section(s: _SummaryStyle, rollup: dict | None) -> list[str]:
    """Box rows for the run's machine-wide resource rollup (peaks/averages, bytes).

    Returns no rows when *rollup* is absent (capture unavailable), so runs
    without resource capture render exactly as before.
    """
    if not rollup:
        return []
    ram = rollup.get("ram_pct") or {}
    cpu = rollup.get("cpu_pct") or {}
    gpu = rollup.get("gpu_pct")
    net_mb = (rollup.get("net_bytes") or 0) / (1024 * 1024)
    disk_mb = (rollup.get("disk_bytes") or 0) / (1024 * 1024)
    lines = [
        s.empty_row(),
        s.mid,
        s.empty_row(),
        s.metric_row1(
            "RAM (machine)",
            f"avg {ram.get('avg', 0):.1f}%  peak {ram.get('peak', 0):.1f}%",
        ),
        s.metric_row1(
            "CPU (machine)",
            f"avg {cpu.get('avg', 0):.1f}%  peak {cpu.get('peak', 0):.1f}%",
        ),
    ]
    if gpu:
        lines.append(
            s.metric_row1(
                "GPU (machine)",
                f"avg {gpu.get('avg', 0):.1f}%  peak {gpu.get('peak', 0):.1f}%",
            )
        )
    lines.append(
        s.metric_row1("NET / DISK (machine)", f"{net_mb:.1f} MB / {disk_mb:.1f} MB")
    )
    return lines


def _subagent_rollup_section(s: _SummaryStyle, rollup: dict | None) -> list[str]:
    """Box rows for task-level cost when delegations occurred: own vs rolled-up
    totals plus one line per direct subagent (session id, tokens, elapsed).

    Returns no rows when *rollup* is absent or carries no descendants, so
    delegation-free runs render exactly as before.
    """
    r = rollup or {}
    descendants = r.get("descendants") or []
    if not descendants:
        return []
    own_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for agg in (r.get("by_kind") or {}).values():
        for key in own_total:
            own_total[key] += agg.get(key, 0)
    grand_total = r.get("grand_total") or {}
    lines = [
        s.empty_row(),
        s.mid,
        s.empty_row(),
        s.metric_row1("TASK TOTAL (own)", f"{own_total.get('total_tokens', 0):,} tok"),
        s.metric_row1(
            "TASK TOTAL (rolled up)", f"{grand_total.get('total_tokens', 0):,} tok"
        ),
        s.metric_row1("SUBAGENTS", str(len(descendants))),
        s.empty_row(),
    ]
    for i, child in enumerate(descendants, start=1):
        child_total = child.get("grand_total") or {}
        report_label = (
            "free report" if child.get("report_source") == "verbatim" else "paid report"
        )
        line_vis = (
            f"  {i}. {child.get('session_id', '?')} — "
            f"{child_total.get('total_tokens', 0):,} tok, "
            f"{child.get('elapsed_s', 0.0):.1f}s, {report_label}"
        )
        pad = s.W - len(line_vis)
        lines.append(
            s.BORDER + "║" + f"  {i}. {s.VALUE}{child.get('session_id', '?')}{s.R} — "
            f"{s.VALUE}{child_total.get('total_tokens', 0):,}{s.R} tok, "
            f"{s.VALUE}{child.get('elapsed_s', 0.0):.1f}{s.R}s, "
            f"{s.VALUE}{report_label}{s.R}" + " " * max(pad, 0) + s.BORDER + "║" + s.R
        )
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
    rollup: dict | None = None,
    resource_rollup: dict | None = None,
) -> None:
    """Render the end-of-run summary box to stderr.

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
        rollup: This agent's usage summary (``AgentNode._usage_summary()``);
            when it carries delegated subagents, a task-level cost section is
            appended (own vs rolled-up totals, one line per subagent).
        resource_rollup: Session-wide machine-wide resource rollup
            (``Recorder.resource_rollup()``); the resource section is omitted
            when ``None`` (capture unavailable), so a run without capture
            renders unchanged.
    """
    s = _SummaryStyle()
    metric_row1 = s.metric_row1

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
        *_token_chart_rows(s, llm_calls),
        s.mid,
        metric_row1("TOOL CALLS", tool_count),
    ]

    lines += _tool_calls_section(s, records)
    lines += _context_resets_section(s, handoffs, context_window)
    lines += _subagent_rollup_section(s, rollup)
    lines += _resource_rollup_section(s, resource_rollup)

    if last_message:
        lines += [s.empty_row(), s.mid, s.empty_row()]
        lines += _markdown_rows(s, last_message)

    lines += [s.empty_row(), s.bottom, ""]
    # Intentional terminal-UI output: render the ANSI box directly to stderr,
    # deliberately bypassing the logger so the box is not prefixed by the
    # ColoredFormatter's "timestamp | LEVEL |" header (CONTRIBUTE.md §31/§36).
    sys.stderr.write("\n".join(lines) + "\n")

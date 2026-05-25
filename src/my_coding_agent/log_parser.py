"""Parse ANSI-colored session log files into structured data."""
import re
from dataclasses import dataclass, field
from pathlib import Path


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


@dataclass
class StepMetrics:
    step_num: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_pct: float
    api_time_s: float = 0.0


@dataclass
class ToolCallRecord:
    step_num: int
    call_id: str
    name: str
    ok: bool
    args_preview: str = ""


@dataclass
class SessionLog:
    session_id: str = ""
    model: str = ""
    context_window: int = 0
    start_time: str = ""
    workspace: str = ""
    steps: list[StepMetrics] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stop_reason: str = "unknown"
    total_steps: int = 0
    max_steps: int = 0
    elapsed_seconds: float = 0.0
    log_path: str = ""

    def to_summary_markdown(self) -> str:
        """Return a compact markdown summary suitable for injecting into an agent prompt."""
        lines = [
            "## Parsed Session Summary",
            "",
            f"- **Session ID:** `{self.session_id}`",
            f"- **Model:** {self.model}",
            f"- **Context window:** {self.context_window:,} tokens",
            f"- **Start time:** {self.start_time}",
            f"- **Workspace:** {self.workspace}",
            f"- **Steps:** {self.total_steps} / {self.max_steps}",
            f"- **Stop reason:** {self.stop_reason}",
            f"- **Elapsed:** {self.elapsed_seconds:.1f}s",
            f"- **Log file:** {self.log_path}",
            "",
        ]

        if self.steps:
            lines += [
                "### Token Usage Per Step",
                "",
                "| Step | Prompt | Completion | Total | Context % | API time (s) |",
                "|------|--------|------------|-------|-----------|--------------|",
            ]
            for s in self.steps:
                ctx = f"**{s.context_pct:.1f}%** ⚠️" if s.context_pct > 100 else f"{s.context_pct:.1f}%"
                lines.append(
                    f"| {s.step_num} | {s.prompt_tokens:,} | {s.completion_tokens:,} "
                    f"| {s.total_tokens:,} | {ctx} | {s.api_time_s:.2f} |"
                )
            lines.append("")

        if self.errors:
            lines += ["### Errors", ""]
            for e in self.errors:
                lines.append(f"- `{e}`")
            lines.append("")

        if self.tool_calls:
            total = len(self.tool_calls)
            ok = sum(1 for t in self.tool_calls if t.ok)
            lines += [
                f"### Tool Calls ({ok}/{total} succeeded)",
                "",
                "| Step | Tool | Status | Args preview |",
                "|------|------|--------|--------------|",
            ]
            for t in self.tool_calls:
                status = "✅" if t.ok else "❌"
                lines.append(f"| {t.step_num} | `{t.name}` | {status} | `{t.args_preview[:80]}` |")
            lines.append("")

        return "\n".join(lines)


def parse_session_log(log_path: str | Path) -> SessionLog:
    """Parse a stderr session log into a structured SessionLog."""
    text = Path(log_path).read_text(errors="replace")
    clean = strip_ansi(text)
    lines = clean.splitlines()

    result = SessionLog(log_path=str(log_path))

    # --- session metadata from banner ---
    # SESSION: ff2a5270d0d0
    m = re.search(r"SESSION[:\s]+([a-f0-9]{10,16})", clean)
    if m:
        result.session_id = m.group(1)

    # MODEL: Qwen3.6-35B-A3B-4bit
    m = re.search(r"MODEL[:\s]+([\w.\-]+)", clean)
    if m:
        result.model = m.group(1)

    # Context window for X: 131072 tokens
    m = re.search(r"Context window for [^:]+:\s*(\d+)", clean)
    if m:
        result.context_window = int(m.group(1))

    # CONTEXT: 131,072  (from banner line)
    if not result.context_window:
        m = re.search(r"CONTEXT[:\s]+([\d,]+)", clean)
        if m:
            result.context_window = int(m.group(1).replace(",", ""))

    # WORKSPACE line — strip trailing box-drawing characters left by ANSI banner
    m = re.search(r"WORKSPACE[:\s]+(/[^\n]+)", clean)
    if m:
        result.workspace = re.sub(r"[\s║╗╔╝╚═╠╣]+$", "", m.group(1)).strip()

    # Start time from first timestamp
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", clean)
    if m:
        result.start_time = m.group(1)

    # --- per-step token metrics ---
    # "Step 3 tokens — prompt: 96892, completion: 536, total: 97428 / 131072 (74.3% used)"
    step_token_re = re.compile(
        r"Step (\d+) tokens — prompt: ([\d,]+), completion: ([\d,]+), total: ([\d,]+)"
        r"(?:[^(]*)?\(([\d.]+)% used\)?"
    )
    # API time from response: "total_time": 226.84
    # keyed by position in log; we'll pair them with steps in order
    api_times = [float(x) for x in re.findall(r'"total_time":\s*([\d.]+)', clean)]

    for i, sm in enumerate(step_token_re.finditer(clean)):
        api_t = api_times[i] if i < len(api_times) else 0.0
        result.steps.append(StepMetrics(
            step_num=int(sm.group(1)),
            prompt_tokens=int(sm.group(2).replace(",", "")),
            completion_tokens=int(sm.group(3).replace(",", "")),
            total_tokens=int(sm.group(4).replace(",", "")),
            context_pct=float(sm.group(5)),
            api_time_s=api_t,
        ))

    # --- errors ---
    for line in lines:
        if "| ERROR |" in line:
            result.errors.append(line.split("| ERROR |", 1)[-1].strip())

    # --- tool calls ---
    # Each dispatch line: "| TOOL | call_xxx → funcname({'arg': ...})"
    # Args may span multiple lines for long bash commands, so we use DOTALL
    # and match from the tool name up to the matching close-paren or end-of-args.
    # Simpler approach: match call_id + func_name from a line, args_preview from same line.
    tool_dispatch_re = re.compile(r"\| TOOL \| (call_\w+) → (\w+)\(")
    # error lines: "error call_xxx → func: message"
    error_call_re = re.compile(r"\| ERROR \| error (call_\w+) →")
    error_ids: set[str] = {m.group(1) for m in error_call_re.finditer(clean)}
    # result lines look like "call_xxx → funcname: stdout:" — skip them
    result_line_re = re.compile(r"\| TOOL \| call_\w+ → \w+: ")

    current_step = 0
    for line in lines:
        sm = re.search(r"STEP (\d+)/(\d+)", line)
        if sm:
            current_step = int(sm.group(1))
            result.max_steps = int(sm.group(2))
            continue

        # skip result lines before checking for dispatch
        if result_line_re.search(line):
            continue

        tm = tool_dispatch_re.search(line)
        if tm:
            call_id = tm.group(1)
            # args preview: everything after "funcname(" on this line, up to 100 chars
            args_start = tm.end()
            args_preview = line[args_start:args_start + 100].rstrip(")")
            result.tool_calls.append(ToolCallRecord(
                step_num=current_step,
                call_id=call_id,
                name=tm.group(2),
                ok=call_id not in error_ids,
                args_preview=args_preview,
            ))

    # --- summary block ---
    # "STEPS: 17 / 20   STOP REASON: stop"
    m = re.search(r"STEPS[:\s]+(\d+)\s*/\s*(\d+)", clean)
    if m:
        result.total_steps = int(m.group(1))
        result.max_steps = int(m.group(2))

    m = re.search(r"STOP REASON[:\s]+(\w+)", clean)
    if m:
        result.stop_reason = m.group(1)

    # ELAPSED from summary
    m = re.search(r"ELAPSED[:\s]+([\d]+m [\d.]+s|[\d.]+s)", clean)
    if m:
        elapsed_raw = m.group(1)
        mm = re.match(r"(\d+)m ([\d.]+)s", elapsed_raw)
        if mm:
            result.elapsed_seconds = int(mm.group(1)) * 60 + float(mm.group(2))
        else:
            result.elapsed_seconds = float(elapsed_raw.rstrip("s"))

    if not result.total_steps and result.steps:
        result.total_steps = len(result.steps)

    return result

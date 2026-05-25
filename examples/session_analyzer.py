"""
Session Analyzer Agent
----------------------
Analyzes the latest (or a specified) session log, researches current harness
engineering best-practices, and writes a structured report to:

    .my_coding_agent/<session-id>/session_analysis.md

Usage:
    uv run python examples/session_analyzer.py
    uv run python examples/session_analyzer.py --log examples/stderr.log
    uv run python examples/session_analyzer.py --log path/to/stderr.log --max-steps 25
"""
import os
import sys
import argparse
from pathlib import Path

# ensure the local package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my_coding_agent import LLM, Agent, tool, ToolsRegistry
from my_coding_agent.log_parser import parse_session_log


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Analyze a my-coding-agent session log")
parser.add_argument(
    "--log", "-l",
    default="examples/stderr.log",
    help="Path to the stderr session log (default: examples/stderr.log)",
)
parser.add_argument(
    "--max-steps", type=int, default=20,
    help="Max agent steps for the analysis run (default: 20)",
)
args = parser.parse_args()


# ── Parse the log ──────────────────────────────────────────────────────────────

log_path = Path(args.log)
if not log_path.exists():
    print(f"Error: log file not found: {log_path}", file=sys.stderr)
    sys.exit(1)

print(f"Parsing session log: {log_path}", file=sys.stderr)
session = parse_session_log(log_path)

if not session.session_id:
    print("Warning: could not extract session ID from log — using 'unknown'", file=sys.stderr)
    session.session_id = "unknown"

print(f"Session ID: {session.session_id}", file=sys.stderr)
print(f"Steps: {session.total_steps}/{session.max_steps}  |  Stop: {session.stop_reason}", file=sys.stderr)
print(f"Errors detected: {len(session.errors)}", file=sys.stderr)

output_dir  = Path(".my_coding_agent") / session.session_id
output_path = output_dir / "session_analysis.md"

# ── Build the structured session summary ──────────────────────────────────────

session_summary = session.to_summary_markdown()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior AI systems engineer specializing in agentic harness design. "
    "Your task is to produce a detailed, actionable session analysis report for a "
    "my-coding-agent run. You have access to the parsed session data, the source code, "
    "and the web to research the latest harness engineering techniques.\n\n"
    "Available tools:\n"
    "- bash(command) — run shell commands\n"
    "- read_file(path) — read any file\n"
    "- write_file(path, content) — write the final report\n"
    "- read_article(url) — fetch a web page as clean markdown\n\n"
    "Workspace context:\n"
    f"- Current path: {os.getcwd()}\n"
    f"- Session log: {log_path.resolve()}\n"
    f"- Output report: {output_path}\n"
    f"- Git recent commits: {os.popen('git log -5 --oneline').read().strip()}\n"
)

# ── User prompt with the parsed session data injected ─────────────────────────

USER_PROMPT = f"""You are analyzing a completed agent session. Here is the structured data extracted from the log:

{session_summary}

Your job is to produce a comprehensive session analysis report and save it to:
`{output_path}`

## Steps to complete this task

### 1. Read source code for context
Read the core agent library files to understand the implementation:
- `src/my_coding_agent/agent.py`
- `src/my_coding_agent/llm.py`
- `src/my_coding_agent/tools.py`

### 2. Research latest harness engineering techniques
Fetch at least one of these references to ground your recommendations in current best practices:
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://www.anthropic.com/research/building-effective-agents

Focus on: context management, multi-agent orchestration, tool reliability, failure recovery, cost optimization.

### 3. Investigate failure patterns in the log
Run grep commands on the log to extract specific failure details:
```bash
grep -n "ERROR\\|WARNING\\|context_limit\\|max_steps" {log_path.resolve()} | grep -v "DEBUG" | head -30
```

Also check the context growth pattern:
```bash
grep "Step [0-9]* tokens" {log_path.resolve()} | sed 's/.*Step/Step/'
```

### 4. Write the analysis report

Save the report to `{output_path}` with this structure:

```markdown
# Session Analysis Report

**Session ID:** <id>
**Generated:** <date>
**Model:** <model>
**Log:** <path>

---

## Executive Summary
2-3 sentences: what happened, did it succeed, the biggest issue.

## Session Metrics
Table: steps, tokens, context %, API times, elapsed.

## Failure Modes Identified
For each failure: what happened, root cause, severity (critical/major/minor), fix.

## Successful Patterns
What worked well that should be preserved.

## Harness Engineering Recommendations
Concrete, prioritized improvements grounded in current best-practices research.
Each recommendation: Problem → Evidence from this session → Recommended fix → Expected impact.

## Code-Level Action Items
Specific files and changes needed (with file paths).

## Version History
| Date | Session | Change |
```

Be specific and concrete. Reference exact step numbers, token counts, and error messages from the parsed data.
Do NOT hallucinate — every finding must be traceable to the session data or the source code you read.
"""

# ── Tools ─────────────────────────────────────────────────────────────────────

tools = [
    tool(ToolsRegistry.bash),
    tool(ToolsRegistry.read_file),
    tool(ToolsRegistry.write_file),
    tool(ToolsRegistry.read_article),
]

# ── Agent ─────────────────────────────────────────────────────────────────────

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": USER_PROMPT},
]

agent = Agent(
    messages=messages,
    tools=tools,
    label="Session Analyzer",
)

agent.run(max_steps=args.max_steps)

# ── Confirm output ─────────────────────────────────────────────────────────────

if output_path.exists():
    size = output_path.stat().st_size
    print(f"\nReport written: {output_path} ({size:,} bytes)", file=sys.stderr)
else:
    print(f"\nWarning: agent did not write {output_path}", file=sys.stderr)

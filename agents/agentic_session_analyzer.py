"""
Session Analyzer Agent
----------------------
Analyzes a session log, researches current harness engineering best-practices,
and writes a structured report to:

    .my_coding_agent/<session-id>/session_analysis.md

Usage (standalone):
    uv run python agents/agentic_session_analyzer.py
    uv run python agents/agentic_session_analyzer.py --log agents/stderr.log
    uv run python agents/agentic_session_analyzer.py --log path/to/stderr.log --max-steps 25

Importable:
    from agents.agentic_session_analyzer import run_analysis
    run_analysis("agents/stderr.log", max_steps=20)
"""
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my_coding_agent import Agent, tool, ToolsRegistry
from my_coding_agent.log_parser import parse_session_log


def run_analysis(log_path: str | Path = "agents/stderr.log", max_steps: int = 20) -> Path | None:
    """Parse *log_path*, run the Session Analyzer agent, return the report path (or None)."""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"[agentic_session_analyzer] log file not found: {log_path}", file=sys.stderr)
        return None

    print(f"[agentic_session_analyzer] parsing log: {log_path}", file=sys.stderr)
    session = parse_session_log(log_path)

    if not session.session_id:
        session.session_id = "unknown"
        print("[agentic_session_analyzer] warning: could not extract session ID — using 'unknown'", file=sys.stderr)

    print(
        f"[agentic_session_analyzer] session={session.session_id}  "
        f"steps={session.total_steps}/{session.max_steps}  "
        f"stop={session.stop_reason}  errors={len(session.errors)}",
        file=sys.stderr,
    )

    output_path = Path(".my_coding_agent") / session.session_id / "session_analysis.md"
    session_summary = session.to_summary_markdown()

    system_prompt = (
        "You are a senior AI systems engineer specialising in agentic harness design. "
        "Produce a detailed, actionable session analysis report for a my-coding-agent run. "
        "You have access to the parsed session data, the source code, and the web.\n\n"
        "Available tools:\n"
        "- bash(command) — run shell commands\n"
        "- read_file(path) — read any file\n"
        "- write_file(path, content) — write the final report\n"
        "- read_article(url) — fetch a web page as clean markdown\n\n"
        f"Current path: {os.getcwd()}\n"
        f"Session log:  {log_path.resolve()}\n"
        f"Output report: {output_path}\n"
        f"Git recent commits:\n{os.popen('git log -5 --oneline').read().strip()}\n"
    )

    user_prompt = f"""You are analysing a completed agent session. Structured data extracted from the log:

{session_summary}

Produce a comprehensive session analysis report and save it to `{output_path}`.

## Steps

### 1. Read source code for context
- `src/my_coding_agent/agent.py`
- `src/my_coding_agent/llm.py`
- `src/my_coding_agent/tools.py`

### 2. Research latest harness engineering best practices
Fetch at least one reference:
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://www.anthropic.com/research/building-effective-agents

### 3. Deep-dive into the log
```bash
grep -n "ERROR\\|WARNING\\|context_limit\\|max_steps" {log_path.resolve()} | grep -v "DEBUG" | head -30
grep "Step [0-9]* tokens" {log_path.resolve()} | sed 's/.*Step/Step/'
```

### 4. Write the report to `{output_path}`

Structure:
```
# Session Analysis Report
**Session ID / Generated / Model / Log**

## Executive Summary
## Session Metrics  (table: step, prompt tok, ctx%, api time)
## Failure Modes Identified  (what / root cause / severity / fix)
## Successful Patterns
## Harness Engineering Recommendations  (problem → evidence → fix → impact)
## Code-Level Action Items  (file paths + specific changes)
## Version History
```

Every finding must be traceable to the session data or source code. No hallucination.
"""

    tools = [
        tool(ToolsRegistry.bash),
        tool(ToolsRegistry.read_file),
        tool(ToolsRegistry.write_file),
        tool(ToolsRegistry.read_article),
    ]

    agent = Agent(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        tools=tools,
        label="Session Analyzer",
    )
    agent.run(max_steps=max_steps)

    if output_path.exists():
        print(f"[agentic_session_analyzer] report → {output_path} ({output_path.stat().st_size:,} bytes)", file=sys.stderr)
        return output_path

    print(f"[agentic_session_analyzer] warning: agent did not write {output_path}", file=sys.stderr)
    return None


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--log", "-l", default="agents/stderr.log", show_default=True,
              type=click.Path(), help="Path to the stderr session log.")
@click.option("--max-steps", default=20, show_default=True,
              type=click.IntRange(1, 100), help="Max agent loop steps.")
def cli(log, max_steps):
    """Run the Session Analyzer Agent.

    Parses a session log, researches harness engineering best-practices,
    and writes a report to .my_coding_agent/<session-id>/session_analysis.md.
    """
    run_analysis(log_path=log, max_steps=max_steps)


if __name__ == "__main__":
    cli()

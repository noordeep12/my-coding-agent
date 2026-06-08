"""
Session Analyzer Agent
-----------------------
Analyzes a completed agent session to identify failure modes, successful patterns,
and recommend harness improvements. Outputs a report to:
  .my_coding_agent/<session-id>/session_analysis.md

Usage (standalone):
    uv run python agents/session_analyzer.py --session-id <id>
    uv run python agents/session_analyzer.py   # auto-picks most recent session

Importable:
    from agents.session_analyzer import run_analysis
    run_analysis(session_id="ff2a5270d0d0")
"""
import json
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my_coding_agent import Agent, tool, ToolsRegistry

_BASE_DIR = Path(".my_coding_agent")


def _most_recent_session() -> str | None:
    """Return session_id of the most recently modified session_data.json, or None."""
    candidates = sorted(
        _BASE_DIR.glob("*/session_data.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parent.name


def _build_system_prompt(tools: list) -> str:
    tool_docs = "\n".join(
        f"  - {t['function']['name']}({', '.join(t['function']['parameters']['properties'].keys())})"
        f" — {t['function']['description']}"
        for t in tools
    )
    is_git = os.path.isdir(".git")
    return (
        "You are a harness improvement specialist. Your job is to analyze completed LLM agent "
        "sessions and produce actionable improvement reports.\n\n"
        f"Available tools:\n{tool_docs}\n\n"
        "Workspace:\n"
        f"  path     : {os.getcwd()}\n"
        f"  contents : {os.listdir(os.getcwd())}\n"
        f"  os       : {os.name}, platform: {sys.platform}, user: {os.getlogin()}\n"
        + (
            f"  git      : {os.popen('git status --short').read().strip() or 'clean'}\n"
            f"  branch   : {os.popen('git rev-parse --abbrev-ref HEAD').read().strip()}\n"
            f"  commits  :\n{os.popen('git log -5 --oneline').read().strip()}\n"
            if is_git else "  git      : not a git repository\n"
        )
        + "\nUse absolute paths when working with files."
    )

_USER_PROMPT_TEMPLATE = """\
Analyze this agent session and produce a structured improvement report.

## Session Data
```json
{session_json}
```

## Your Tasks

1. **Skim the session log file** at `.my_coding_agent/{session_id}/stderr.log`.
   The log can be very large — DO NOT read it whole. Use targeted bash commands to extract signal:

   ```bash
   # Size and line count
   wc -l .my_coding_agent/{session_id}/stderr.log
   du -sh .my_coding_agent/{session_id}/stderr.log

   # Errors and warnings (with 2 lines of context each)
   grep -n "| ERROR\|| WARNING\|| CRITICAL" .my_coding_agent/{session_id}/stderr.log | head -60

   # Tool dispatches (shows which tools were called and their args)
   grep -n "tool_id\|→" .my_coding_agent/{session_id}/stderr.log | head -80

   # Step markers (one per agent loop iteration)
   grep -n "STEP [0-9]" .my_coding_agent/{session_id}/stderr.log

   # Context % per step (token pressure over time)
   grep -n "% ctx used\|Context at\|Context reset\|Context limit" .my_coding_agent/{session_id}/stderr.log

   # Handoff events
   grep -n "handoff\|Handoff\|context reset\|continuation" .my_coding_agent/{session_id}/stderr.log | head -30

   # Last 80 lines (final agent state, summary output)
   tail -n 80 .my_coding_agent/{session_id}/stderr.log
   ```

   Use these outputs — not the raw file — as your log evidence.

2. **Read the harness source code** to understand the current implementation:
   - `src/my_coding_agent/agent.py` — agent loop, context handling
   - `src/my_coding_agent/llm.py` — LLM client, tool execution
   - `src/my_coding_agent/tools.py` — available tools

3. **Fetch one authoritative reference** on LLM agent harness best practices. Use read_article to fetch:
   https://www.anthropic.com/engineering/building-effective-agents

4. **Analyze the session** for:
   - Failure modes (tool errors, context issues, wrong stop reasons, repeated retries, etc.)
   - Successful patterns worth preserving
   - Token efficiency (prompt vs completion ratio, context % at stop)
   - Context reset events (if any)
   - Any unusual patterns visible in the log (long silences, burst tool calls, error cascades)

5. **Write the report** to `.my_coding_agent/{session_id}/session_analysis.md` using write_file.

The report MUST follow this exact structure:
```markdown
# Session Analysis: {session_id}

## Session Metrics
- Agent: <label>
- Model: <model>
- Started: <started_at>
- Elapsed: <elapsed_seconds>s
- Steps: <steps> / <max_steps> (stop reason: <stop_reason>)
- Tokens: prompt=<prompt_tokens>, completion=<completion_tokens>, total=<total_tokens>
- Context: <context_window> tokens (<pct>% used at stop)
- Context resets: <N>
- Tool calls: <total> total, <ok> ok, <failed> failed

## Failure Mode Analysis
<!-- Each identified failure: what happened, why, impact -->

## Successful Patterns
<!-- What worked well, should be preserved or replicated -->

## Recommended Improvements
<!-- Numbered, specific, actionable — code changes, prompt changes, config changes -->

## References
<!-- Sources consulted -->
```

Write only facts derived from the session data and source code. Be specific and actionable.
"""


def run_analysis(session_id: str | None = None, max_steps: int = 15) -> Path | None:
    """Run the Session Analyzer Agent and return the path to session_analysis.md."""
    if session_id is None:
        session_id = _most_recent_session()
        if session_id is None:
            print("[session-analyzer] No sessions found in .my_coding_agent/", file=sys.stderr)
            return None
        print(f"[session-analyzer] Auto-selected most recent session: {session_id}", file=sys.stderr)

    session_data_path = _BASE_DIR / session_id / "session_data.json"
    if not session_data_path.exists():
        print(f"[session-analyzer] Session data not found: {session_data_path}", file=sys.stderr)
        return None

    session_json = session_data_path.read_text()
    session_data = json.loads(session_json)

    print(f"[session-analyzer] Analyzing session {session_id} ({session_data.get('steps', '?')} steps, "
          f"stop={session_data.get('stop_reason', '?')})", file=sys.stderr)

    tools = [
        tool(ToolsRegistry.bash),
        tool(ToolsRegistry.read_file),
        tool(ToolsRegistry.write_file),
        tool(ToolsRegistry.read_article),
        tool(ToolsRegistry.read_tool_artifact),
    ]

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        session_json=session_json,
        session_id=session_id,
    )

    agent = Agent(
        messages=[
            {"role": "system", "content": _build_system_prompt(tools)},
            {"role": "user",   "content": user_prompt},
        ],
        tools=tools,
        label="Session Analyzer",
    )
    agent.run(max_steps=max_steps)

    out = _BASE_DIR / session_id / "session_analysis.md"
    if out.exists():
        print(f"[session-analyzer] Report written → {out.resolve()} ({out.stat().st_size:,} bytes)", file=sys.stderr)
        return out

    print(f"[session-analyzer] Warning: agent did not write {out}", file=sys.stderr)
    return None


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--session-id", "-s", default=None, metavar="ID",
              help="Session ID to analyze. Defaults to the most recent session.")
@click.option("--max-steps", default=15, show_default=True,
              type=click.IntRange(1, 100), help="Max agent loop steps.")
def cli(session_id, max_steps):
    """Run the Session Analyzer Agent.

    Reads a completed session's data, analyzes failure modes and patterns,
    and writes a report to .my_coding_agent/<session-id>/session_analysis.md.
    """
    run_analysis(session_id=session_id, max_steps=max_steps)


if __name__ == "__main__":
    cli()

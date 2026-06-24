"""
Session Analyzer Agent
-----------------------
Analyzes a completed agent session to identify failure modes, successful patterns,
and recommend harness improvements. Outputs a report to
``.my_coding_agent/<session-id>/session_analysis.md``.

Usage (standalone)::

    uv run python -m my_coding_agent.agents.session_analyzer --session-id <id>
    uv run python -m my_coding_agent.agents.session_analyzer   # most recent session

Importable::

    from my_coding_agent.agents.session_analyzer import run_analysis
    run_analysis(session_id="ff2a5270d0d0")
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import click

from my_coding_agent import Agent, ToolRegistry, tool
from my_coding_agent.logger import get_logger

logger = get_logger(__name__)

_BASE_DIR = Path(".my_coding_agent")


def _git(*args: str) -> str:
    """Run a git command (no shell) and return stripped stdout, or '' on failure."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True)
    except (OSError, FileNotFoundError):
        return ""
    return result.stdout.strip()


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
        f"  - {t['function']['name']}"
        f"({', '.join(t['function']['parameters']['properties'].keys())})"
        f" — {t['function']['description']}"
        for t in tools
    )
    is_git = os.path.isdir(".git")
    return (
        "You are an AI Harness improvement specialist. Your job is to analyze "
        "completed LLM agent "
        "sessions, list problems, and produce actionable improvement reports.\n\n"
        f"Available tools:\n{tool_docs}\n\n"
        "Workspace:\n"
        f"  current directory     : {os.getcwd()}\n"
        f"  current directory contents : {os.listdir(os.getcwd())}\n"
        f"  machine os       : {os.name}, platform: {sys.platform}, "
        f"user: {os.getlogin()}\n"
        + (
            f"  git      : {_git('status', '--short') or 'clean'}\n"
            f"  branch   : {_git('rev-parse', '--abbrev-ref', 'HEAD')}\n"
            f"  commits  :\n{_git('log', '-5', '--oneline')}\n"
            if is_git
            else "  git      : not a git repository\n"
        )
        + "\nUse absolute paths when tool arguments requires files paths."
    )


_USER_PROMPT_TEMPLATE = """\
Analyze this agent session, list problems, and produce a structured improvement report.

## Session Data
```json
{session_json}
```

## Your Tasks

1. **Skim the session log file** at `.my_coding_agent/{session_id}/stderr.log`.
   The log can be very large — DO NOT read it whole. Use targeted bash
   commands to extract signal:

   ```bash
   # Size and line count
   wc -l .my_coding_agent/{session_id}/stderr.log
   du -sh .my_coding_agent/{session_id}/stderr.log

   # Errors and warnings (with 2 lines of context each)
   grep -n "| ERROR\\|| WARNING\\|| CRITICAL" \
       .my_coding_agent/{session_id}/stderr.log | head -60

   # Tool dispatches (shows which tools were called and their args)
   grep -n "tool_id\\|→" .my_coding_agent/{session_id}/stderr.log | head -80

   # Step markers (one per agent loop iteration)
   grep -n "STEP [0-9]" .my_coding_agent/{session_id}/stderr.log

   # Context % per step (token pressure over time)
   grep -n "% ctx used\\|Context at\\|Context reset\\|Context limit" \
       .my_coding_agent/{session_id}/stderr.log

   # Handoff events
   grep -n "handoff\\|Handoff\\|context reset\\|continuation" \
       .my_coding_agent/{session_id}/stderr.log | head -30

   # Last 80 lines (final agent state, summary output)
   tail -n 80 .my_coding_agent/{session_id}/stderr.log
   ```

   Use these outputs — not the raw file — as your log evidence.

2. **Read the harness source code** to understand the current implementation:
   - `src/my_coding_agent/agent.py` — agent loop, context handling
   - `src/my_coding_agent/llm.py` — LLM client, tool execution
   - `src/my_coding_agent/tools.py` — available tools

3. **Fetch one authoritative reference** on LLM agent harness best practices.
   Use read_article to fetch:
   - https://www.anthropic.com/engineering/building-effective-agents
   - https://medium.com/@tort_mario/ai-agent-best-practices-production-ready-harness-engineering-2026-guide-c1236d713fac
   - https://gist.github.com/celesteanders/21edad2367c8ede2ff092bd87e56a26f
   - https://gist.github.com/amazingvince/52158d00fb8b3ba1b8476bc62bb562e3


4. **Analyze the session** for:
   - Failure modes (tool errors, context issues, wrong stop reasons,
     repeated retries, etc.)
   - Successful patterns worth preserving
   - Token efficiency (prompt vs completion ratio, context % at stop)
   - Any harness events like: context reset, handoff, continuation,
     summarization, tool arg correction, tool routing decisions, etc.
   - Any unusual patterns visible in the log (long silences, burst tool calls,
     error cascades)

5. **Write the report** to `.my_coding_agent/{session_id}/session_analysis.md`
   using write_file.

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

Write only facts derived from the session data and source code.
Be specific and actionable.
"""


def run_analysis(session_id: str | None = None, max_steps: int = 15) -> Path | None:
    """Run the Session Analyzer Agent and return the path to session_analysis.md."""
    if session_id is None:
        session_id = _most_recent_session()
        if session_id is None:
            logger.warning(
                "[session-analyzer] No sessions found in .my_coding_agent/",
            )
            return None
        logger.info(
            "[session-analyzer] Auto-selected most recent session: %s",
            session_id,
        )

    session_data_path = _BASE_DIR / session_id / "session_data.json"
    if not session_data_path.exists():
        logger.warning(
            "[session-analyzer] Session data not found: %s",
            session_data_path,
        )
        return None

    session_json = session_data_path.read_text()
    session_data = json.loads(session_json)

    logger.info(
        "[session-analyzer] Analyzing session %s (%s steps, stop=%s)",
        session_id,
        session_data.get("steps", "?"),
        session_data.get("stop_reason", "?"),
    )

    tools = [
        tool(ToolRegistry.bash),
        tool(ToolRegistry.read_file),
        tool(ToolRegistry.write_file),
        tool(ToolRegistry.read_article),
        tool(ToolRegistry.read_tool_artifact),
    ]

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        session_json=session_json,
        session_id=session_id,
    )

    agent = Agent(
        messages=[
            {"role": "system", "content": _build_system_prompt(tools)},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        label="Session Analyzer",
    )
    agent.run(max_steps=max_steps)

    out = _BASE_DIR / session_id / "session_analysis.md"
    if out.exists():
        logger.info(
            "[session-analyzer] Report written → %s (%s bytes)",
            out.resolve(),
            f"{out.stat().st_size:,}",
        )
        return out

    logger.warning("[session-analyzer] Warning: agent did not write %s", out)
    return None


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--session-id",
    "-s",
    default=None,
    metavar="ID",
    help="Session ID to analyze. Defaults to the most recent session.",
)
@click.option(
    "--max-steps",
    default=15,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Max agent loop steps.",
)
def cli(session_id: str | None, max_steps: int) -> None:
    """Run the Session Analyzer Agent.

    Reads a completed session's data, analyzes failure modes and patterns,
    and writes a report to .my_coding_agent/<session-id>/session_analysis.md.
    """
    run_analysis(session_id=session_id, max_steps=max_steps)


if __name__ == "__main__":
    cli()

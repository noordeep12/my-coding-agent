#!/usr/bin/env python3
"""
Main workflow
=============
Discovery Agent → Main Agent → (optional) Session Analyzer

Run:
    uv run python workflows/main.py [OPTIONS]
    uv run python workflows/main.py --help
"""
import os
import sys
from pathlib import Path

import click

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "agents"))

from my_coding_agent import Agent, tool, ToolsRegistry  # noqa: E402
from agentic_discovery import run_discovery             # noqa: E402
from agentic_session_analyzer import run_analysis       # noqa: E402


_DEFAULT_PROMPT = (
    "Using `git` and `gh` CLI tools, ensure the latest local code changes "
    "are committed and pushed to GitHub with standardized commit messages."
)


def _system_prompt(tools: list) -> str:
    tool_docs = "\n".join(
        f"  - {t['function']['name']}("
        + ", ".join(t["function"]["parameters"]["properties"].keys())
        + f") — {t['function']['description']}"
        for t in tools
    )
    git_root = _ROOT
    return (
        "You are a helpful coding assistant. Use tools when needed. "
        "Use absolute paths when working with files. Running on macOS.\n\n"
        f"Available tools:\n{tool_docs}\n\n"
        "Workspace:\n"
        f"  path     : {os.getcwd()}\n"
        f"  contents : {os.listdir(os.getcwd())}\n"
        f"  git      : {os.popen('git status --short').read().strip() or 'clean'}\n"
        f"  branch   : {os.popen('git rev-parse --abbrev-ref HEAD').read().strip()}\n"
        f"  commits  :\n{os.popen('git log -5 --oneline').read().strip()}\n"
        + (
            f"\nDiscovery notes:\n"
            + Path(".my_coding_agent/discovery.md").read_text()
            if Path(".my_coding_agent/discovery.md").exists()
            else "\n(no discovery notes — run with --discover to generate)"
        )
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--prompt", "-p",
    default=None,
    metavar="TEXT",
    help="Task for the Main Agent. Omit to use the default commit-and-push task.",
)
@click.option(
    "--interactive", "-i",
    is_flag=True,
    help="Read the task prompt interactively from stdin (paste freely, Ctrl+D to submit).",
)
@click.option(
    "--discover", "-d",
    is_flag=True,
    help="Force-refresh .my_coding_agent/discovery.md before running the Main Agent.",
)
@click.option(
    "--analyze", "-a",
    is_flag=True,
    help="Run the Session Analyzer after the Main Agent finishes.",
)
@click.option(
    "--analyze-log",
    default="agents/stderr.log",
    show_default=True,
    type=click.Path(),
    help="Session log for the Session Analyzer.",
)
@click.option(
    "--max-steps",
    default=20,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Maximum agent loop steps for the Main Agent.",
)
@click.version_option(version="0.1.0", prog_name="my-coding-agent")
def main(prompt, interactive, discover, analyze, analyze_log, max_steps):
    """Run the full coding-agent workflow.

    \b
    Steps executed:
      1. Discovery Agent  — maps the workspace (skipped if discovery.md exists,
                            unless --discover is set)
      2. Main Agent       — executes the requested task
      3. Session Analyzer — analyses the run log (only with --analyze)

    \b
    Examples:
      uv run python workflows/main.py
      uv run python workflows/main.py -p "write tests for llm.py"
      uv run python workflows/main.py --discover --analyze
      uv run python workflows/main.py -i          # paste a multi-line prompt
    """
    # ── resolve prompt ─────────────────────────────────────────────────────────
    if interactive:
        click.secho("Enter your prompt (Ctrl+D when done):", fg="cyan")
        click.echo("─" * 60)
        user_prompt = click.get_text_stream("stdin").read().strip()
        click.echo("─" * 60)
        if not user_prompt:
            click.secho("No prompt entered — using default.", fg="yellow", err=True)
            user_prompt = _DEFAULT_PROMPT
    elif prompt:
        user_prompt = prompt
    else:
        user_prompt = _DEFAULT_PROMPT

    # ── step 1: discovery ──────────────────────────────────────────────────────
    click.secho("\n● Discovery Agent", fg="cyan", bold=True, err=True)
    run_discovery(force=discover)

    # ── step 2: main agent ─────────────────────────────────────────────────────
    click.secho("\n● Main Agent", fg="cyan", bold=True, err=True)
    tools = [
        tool(ToolsRegistry.bash),
        tool(ToolsRegistry.read_file),
        tool(ToolsRegistry.write_file),
    ]
    agent = Agent(
        messages=[
            {"role": "system", "content": _system_prompt(tools)},
            {"role": "user",   "content": user_prompt},
        ],
        tools=tools,
        label="Main Agent",
    )
    agent.run(max_steps=max_steps)

    # ── step 3: session analyzer (optional) ───────────────────────────────────
    if analyze:
        click.secho("\n● Session Analyzer", fg="cyan", bold=True, err=True)
        if not Path(analyze_log).exists():
            click.secho(
                f"  Log not found: {analyze_log}\n"
                "  Tip: redirect stderr when running — e.g.\n"
                "       uv run python workflows/main.py --analyze 2> agents/stderr.log",
                fg="yellow", err=True,
            )
        else:
            run_analysis(log_path=analyze_log, max_steps=20)


if __name__ == "__main__":
    main()

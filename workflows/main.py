#!/usr/bin/env python3
"""
Main workflow
=============
Discovery Agent → Main Agent

Run:
    uv run python workflows/main.py [OPTIONS]
    uv run python workflows/main.py --help
"""
import os
from pathlib import Path

import inspect

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from my_coding_agent import Agent, tool, ToolsRegistry
from agents.discovery import run_discovery
from agents.session_analyzer import run_analysis


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
            "\nDiscovery notes are available at `.my_coding_agent/discovery.md` — "
            "use the read_file tool to consult them when you need codebase context."
            if Path(".my_coding_agent/discovery.md").exists()
            else "\n(no discovery notes — run with --discover to generate)"
        )
    )

_HISTORY_FILE = Path.home() / ".my_coding_agent_history"


def _all_tools() -> list:
    names = [
        name for name, _ in inspect.getmembers(ToolsRegistry, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    return [tool(getattr(ToolsRegistry, name)) for name in names]


def _read_interactive_prompt() -> str:
    """Rich interactive prompt with history, cursor navigation, and multi-line input.

    Keybindings:
      Enter          — new line
      Meta+Enter / Escape then Enter — submit
      Up / Down      — cycle through previous prompts
      Ctrl+C         — cancel (returns empty string)
    """
    kb = KeyBindings()

    @kb.add(Keys.ControlC)
    def _cancel(event):
        event.app.exit(result="")

    @kb.add(Keys.Escape, Keys.ControlM)   # Escape then Enter
    @kb.add("escape", "enter")
    def _submit_esc_enter(event):
        event.current_buffer.validate_and_handle()

    session: PromptSession = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        key_bindings=kb,
        multiline=True,
        prompt_continuation=lambda width, line_number, wrap_count: "  " + "·" * (width - 2),
        enable_history_search=True,
    )

    click.secho("Enter your prompt  (Meta+Enter or Esc→Enter to submit, ↑/↓ for history):", fg="cyan")
    click.echo("─" * 60)
    try:
        text = session.prompt("❯ ")
    except (EOFError, KeyboardInterrupt):
        text = ""
    click.echo("─" * 60)
    return text.strip()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--prompt", "-p",
    default=None,
    show_default="default commit-and-push task",
    metavar="TEXT",
    help="Task for the Main Agent.",
)
@click.option(
    "--interactive", "-i",
    default=True,
    is_flag=True,
    show_default=True,
    help="Read the task prompt interactively from stdin (paste freely, Ctrl+D to submit).",
)
@click.option(
    "--discover/--no-discover", "-d/-D",
    default=False,
    show_default=True,
    help="Run the Discovery Agent before the Main Agent.",
)
@click.option(
    "--max-steps",
    default=20,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Maximum agent loop steps for the Main Agent.",
)
@click.option(
    "--analyze/--no-analyze", "-a/-A",
    default=False,
    show_default=True,
    help="Run the Session Analyzer Agent after the Main Agent completes.",
)
@click.version_option(version="0.1.0", prog_name="my-coding-agent")
def main(prompt, interactive, discover, max_steps, analyze):
    """Run the full coding-agent workflow.

    \b
    Steps executed:
      1. Discovery Agent  — maps the workspace (skip with --no-discover)
      2. Main Agent       — executes the requested task
      3. Session Analyzer — reviews the session and writes a report (opt-in with --analyze)

    \b
    Examples:
      uv run python workflows/main.py
      uv run python workflows/main.py -p "write tests for llm.py"
      uv run python workflows/main.py --no-discover
      uv run python workflows/main.py -i          # paste a multi-line prompt
      uv run python workflows/main.py --analyze   # also run session analysis
    """
    # ── resolve prompt ─────────────────────────────────────────────────────────
    if interactive:
        user_prompt = _read_interactive_prompt()
        if not user_prompt:
            click.secho("No prompt entered — using default.", fg="yellow", err=True)
            user_prompt = _DEFAULT_PROMPT
    elif prompt:
        user_prompt = prompt
    else:
        user_prompt = _DEFAULT_PROMPT

    # ── step 1: discovery ──────────────────────────────────────────────────────
    if discover:
        click.secho("\n● Discovery Agent", fg="cyan", bold=True, err=True)
        run_discovery(force=True)
    else:
        click.secho("\n● Discovery Agent  (skipped via --no-discover)", fg="yellow", err=True)

    # ── step 2: main agent ─────────────────────────────────────────────────────
    click.secho("\n● Main Agent", fg="cyan", bold=True, err=True)
    tools = _all_tools()

    agent = Agent(
        messages=[
            {"role": "system", "content": _system_prompt(tools)},
            {"role": "user",   "content": user_prompt},
        ],
        tools=tools,
        label="Main Agent",
    )
    agent.run(max_steps=max_steps)

    # ── step 3: session analysis (optional) ───────────────────────────────────
    if analyze:
        click.secho("\n● Session Analyzer", fg="cyan", bold=True, err=True)
        run_analysis(session_id=agent.session_id)


if __name__ == "__main__":
    main()

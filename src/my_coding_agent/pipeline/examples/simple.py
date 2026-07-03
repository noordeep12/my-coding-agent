#!/usr/bin/env python3
"""Simple coding-agent pipeline entry point.

Run::

    uv run my-coding-agent [OPTIONS]
    uv run my-coding-agent --help
"""

import inspect
import os
import platform
from datetime import datetime
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys

from my_coding_agent import (
    DEFAULT_MAX_STEPS,
    AgentNode,
    ToolRegistry,
    __version__,
    tool,
)
from my_coding_agent.engine import OMLX_MODEL

_DEFAULT_PROMPT = (
    "Using `git` and `gh` CLI tools, ensure the latest local code changes "
    "are committed and pushed to GitHub with standardized commit messages."
)


def _system_prompt() -> str:
    cwd = os.getcwd()
    is_git_repo = os.path.isdir(os.path.join(cwd, ".git"))
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M (%Z)")
    return (
        "You are a helpful coding assistant working in a terminal. Use tools to "
        "get things done.\n\n"
        "Tool usage:\n"
        "- Use absolute paths when working with files.\n"
        "- When several tool calls are independent, emit them together in one "
        "assistant message.\n"
        "- Every tool returns JSON: "
        '{"schema_version", "tool", "ok", "output", "error", "metadata"}. '
        "When `ok` is true read `output`; when false read `error` (and "
        "`metadata`, e.g. `exit_code`) to recover.\n\n"
        "Communication:\n"
        "- Lead with the outcome: the first sentence of a final answer states "
        "what happened or what you found. Be concise.\n\n"
        "Safety:\n"
        "- Ask the user before destructive operations (force-push, branch "
        "deletion, rm -rf, history rewrites).\n"
        "- Never stage or commit secrets: .env files, keys, tokens, "
        "certificates, credentials.\n"
        "- Commit messages use Conventional Commits: type(scope): description, "
        "<=72 chars, present tense.\n"
        "- Do only what the task asks; do not modify unrelated files.\n\n"
        "If the context window nears its limit, the harness summarizes and "
        "hands off automatically — work normally until then.\n\n"
        "Environment:\n"
        f"- Working directory: {cwd} (git repository: "
        f"{'true' if is_git_repo else 'false'})\n"
        f"- Platform: {platform.system()} {platform.release()}; "
        f"shell: {os.environ.get('SHELL', 'unknown')}\n"
        f"- Model: {OMLX_MODEL}\n\n"
        f"Current date and time: {now}"
    )


_HISTORY_FILE = Path.home() / ".my_coding_agent_history"


def _all_tools() -> list:
    names = [
        name
        for name, _ in inspect.getmembers(ToolRegistry, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    return [tool(getattr(ToolRegistry, name)) for name in names]


def _read_interactive_prompt() -> str:
    """Read a multi-line prompt with history and key bindings."""
    kb = KeyBindings()

    @kb.add(Keys.ControlC)
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result="")

    @kb.add(Keys.Escape, Keys.ControlM)
    @kb.add("escape", "enter")
    def _submit_esc_enter(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    session: PromptSession = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        key_bindings=kb,
        multiline=True,
        prompt_continuation=lambda width, line_number, wrap_count: (
            "  " + "·" * (width - 2)
        ),
        enable_history_search=True,
    )

    click.secho(
        "Enter your prompt  (Meta+Enter or Esc→Enter to submit, ↑/↓ for history):",
        fg="cyan",
    )
    click.echo("─" * 60)
    try:
        text: str = session.prompt("❯ ")
    except (EOFError, KeyboardInterrupt):
        text = ""
    click.echo("─" * 60)
    return text.strip()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--prompt",
    "-p",
    default=None,
    metavar="TEXT",
    help="Task for the agent.",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Read the task prompt interactively.",
)
@click.option(
    "--max-steps",
    default=DEFAULT_MAX_STEPS,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Maximum agent loop steps.",
)
@click.version_option(version=__version__, prog_name="my-coding-agent")
def main(prompt: str | None, interactive: bool, max_steps: int) -> None:
    """Run the coding-agent pipeline.

    \b
    Examples:
      uv run my-coding-agent
      uv run my-coding-agent -p "write tests for llm.py"
      uv run my-coding-agent -i
    """
    if interactive:
        user_prompt = _read_interactive_prompt()
        if not user_prompt:
            click.secho("No prompt entered — using default.", fg="yellow", err=True)
            user_prompt = _DEFAULT_PROMPT
    elif prompt:
        user_prompt = prompt
    else:
        user_prompt = _DEFAULT_PROMPT

    tools = _all_tools()
    agent = AgentNode(
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        label="Main Agent",
    )
    agent.execute(max_steps=max_steps)


if __name__ == "__main__":
    main()

"""
Discovery Agent
---------------
Explores the workspace and writes a stable context document to
.my_coding_agent/discovery.md for use as a system-prompt prefix in future runs.

Usage (standalone)::

    uv run python -m my_coding_agent.agents.discovery
    uv run python -m my_coding_agent.agents.discovery --force   # overwrite existing

Importable::

    from my_coding_agent.agents.discovery import run_discovery
    run_discovery(force=True)
"""

import os
import subprocess
import sys
from pathlib import Path

import click

from my_coding_agent import Agent, ToolsRegistry, tool
from my_coding_agent.logger import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = ".my_coding_agent/discovery.md"


def _git(*args: str) -> str:
    """Run a git command (no shell) and return stripped stdout, or '' on failure."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True)
    except (OSError, FileNotFoundError):
        return ""
    return result.stdout.strip()


_DISCOVERY_USER_PROMPT = (
    "Explore the workspace and discover any relevant information that can help you "
    "understand the current state of the codebase, git repository, recent activities, "
    "and anything else that might be useful for future tasks. "
    "Summarise your findings into a Markdown file saved to "
    f"`{OUTPUT_PATH}` in the current workspace. "
    "It should contain:\n"
    "1. A summary of the current state of the codebase and git repository\n"
    "2. Any recent activities or changes that might be relevant\n"
    "3. Insights or observations that help understand the context better\n"
    "4. Tool descriptions and how to use them\n"
    "5. Latest user requests\n"
    "6. A recent transcript of the conversation\n\n"
    "The goal is to create a stable prompt prefix that can be prepended to future "
    "conversations to provide context and continuity, even if the conversation history "
    "gets truncated due to token limits."
)


def run_discovery(force: bool = False, max_steps: int = 20) -> Path | None:
    """Run the Discovery Agent and return the path to discovery.md (or None on skip)."""
    out = Path(OUTPUT_PATH)

    if out.exists() and not force:
        logger.info(
            "[discovery] %s already exists — skipping (use force=True to overwrite)",
            OUTPUT_PATH,
        )
        return out

    reason = "forced" if force else "file missing"
    logger.info("[discovery] running Discovery Agent (%s)", reason)

    tools = [
        tool(ToolsRegistry.bash),
        tool(ToolsRegistry.read_file),
        tool(ToolsRegistry.write_file),
        tool(ToolsRegistry.read_article),
        tool(ToolsRegistry.read_tool_artifact),
    ]

    tool_docs = "\n".join(
        f"- {t['function']['name']}"
        f"({', '.join(t['function']['parameters']['properties'].keys())}) "
        f"— {t['function']['description']}"
        for t in tools
    )

    is_git = os.path.isdir(".git")
    git_status = _git("status") if is_git else "Not a git repository"
    git_branch = _git("rev-parse", "--abbrev-ref", "HEAD") if is_git else "N/A"
    git_commits = _git("log", "-5", "--oneline") if is_git else "N/A"
    system_prompt = (
        "You are a helpful assistant. Use tools when needed. "
        "Use absolute paths when working with files. "
        "You are running on a MacBook Pro.\n\n"
        f"Available tools:\n{tool_docs}\n\n"
        "Workspace:\n"
        f"- Current path: {os.getcwd()}\n"
        f"- Directory contents: {os.listdir(os.getcwd())}\n"
        f"- OS: {os.name}, Platform: {sys.platform}, User: {os.getlogin()}\n"
        f"- Git status: {git_status}\n"
        f"- Git branch: {git_branch}\n"
        f"- Git recent commits:\n{git_commits}\n"
    )

    agent = Agent(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _DISCOVERY_USER_PROMPT},
        ],
        tools=tools,
        label="Discovery Agent",
    )
    agent.run(max_steps=max_steps)

    if out.exists():
        logger.info(
            "[discovery] discovery.md written → %s (%s bytes)",
            out.resolve(),
            f"{out.stat().st_size:,}",
        )
        return out

    logger.warning("[discovery] warning: agent did not write %s", OUTPUT_PATH)
    return None


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Overwrite existing discovery.md.")
@click.option(
    "--max-steps",
    default=20,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Max agent loop steps.",
)
def cli(force: bool, max_steps: int) -> None:
    """Run the Discovery Agent.

    Explores the workspace and writes a stable context document to
    .my_coding_agent/discovery.md for use as a system-prompt prefix.
    """
    run_discovery(force=force, max_steps=max_steps)


if __name__ == "__main__":
    cli()

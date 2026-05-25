"""
Discovery Agent
---------------
Explores the workspace and writes a stable context document to
.my_coding_agent/discovery.md for use as a system-prompt prefix in future runs.

Usage (standalone):
    uv run python agents/discovery.py
    uv run python agents/discovery.py --force   # overwrite existing

Importable:
    from agents.discovery import run_discovery
    run_discovery(force=True)
"""
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my_coding_agent import Agent, tool, ToolsRegistry

OUTPUT_PATH = ".my_coding_agent/discovery.md"

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
        print(f"[discovery] {OUTPUT_PATH} already exists — skipping (use force=True to overwrite)", file=sys.stderr)
        return out

    reason = "forced" if force else "file missing"
    print(f"[discovery] running Discovery Agent ({reason})", file=sys.stderr)

    tools = [
        tool(ToolsRegistry.bash),
        tool(ToolsRegistry.read_file),
        tool(ToolsRegistry.write_file),
        tool(ToolsRegistry.read_article),
    ]

    tool_docs = "\n".join(
        f"- {t['function']['name']}({', '.join(t['function']['parameters']['properties'].keys())}) "
        f"— {t['function']['description']}"
        for t in tools
    )

    system_prompt = (
        "You are a helpful assistant. Use tools when needed. Use absolute paths when working with files. "
        "You are running on a MacBook Pro.\n\n"
        f"Available tools:\n{tool_docs}\n\n"
        "Workspace:\n"
        f"- Current path: {os.getcwd()}\n"
        f"- Directory contents: {os.listdir(os.getcwd())}\n"
        f"- OS: {os.name}, Platform: {sys.platform}, User: {os.getlogin()}\n"
        f"- Git status: {os.popen('git status').read().strip() if os.path.isdir('.git') else 'Not a git repository'}\n"
        f"- Git branch: {os.popen('git rev-parse --abbrev-ref HEAD').read().strip() if os.path.isdir('.git') else 'N/A'}\n"
        f"- Git recent commits:\n{os.popen('git log -5 --oneline').read().strip() if os.path.isdir('.git') else 'N/A'}\n"
    )

    agent = Agent(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": _DISCOVERY_USER_PROMPT},
        ],
        tools=tools,
        label="Discovery Agent",
    )
    agent.run(max_steps=max_steps)

    if out.exists():
        print(f"[discovery] discovery.md written → {out.resolve()} ({out.stat().st_size:,} bytes)", file=sys.stderr)
        return out

    print(f"[discovery] warning: agent did not write {OUTPUT_PATH}", file=sys.stderr)
    return None


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Overwrite existing discovery.md.")
@click.option("--max-steps", default=20, show_default=True,
              type=click.IntRange(1, 100), help="Max agent loop steps.")
def cli(force, max_steps):
    """Run the Discovery Agent.

    Explores the workspace and writes a stable context document to
    .my_coding_agent/discovery.md for use as a system-prompt prefix.
    """
    run_discovery(force=force, max_steps=max_steps)


if __name__ == "__main__":
    cli()

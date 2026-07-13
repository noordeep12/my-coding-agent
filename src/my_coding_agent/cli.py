#!/usr/bin/env python3
"""The `my-coding-agent` console-script entry point.

Run::

    uv run my-coding-agent [OPTIONS]
    uv run my-coding-agent --help
"""

import inspect
import os
import platform
import sys
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
from my_coding_agent.engine import OMLX_MODEL, egress, exfil, sandbox
from my_coding_agent.engine.checkpoint import (
    CheckpointError,
    find_last_resumable,
    load_checkpoint,
)
from my_coding_agent.engine.tool_execution import policy
from my_coding_agent.engine.tool_registry import discover_skills
from my_coding_agent.evals.reporting import render_verdict
from my_coding_agent.evals.results import RESULTS_ROOT
from my_coding_agent.evals.run_config_file import (
    ConfigValidationError,
    execute_from_config,
)

# ``use_skill`` is registered conditionally — only when skills are discovered —
# so tool schemas stay byte-identical to today for a skill-free run (D5). It is
# therefore excluded from the automatic public-method scan in ``_all_tools``.
_SPECIAL_TOOLS = {"use_skill"}

# Root under which each run's session directory (and its checkpoint) lives.
_SESSIONS_DIR = Path(".my_coding_agent")

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
        if not name.startswith("_") and name not in _SPECIAL_TOOLS
    ]
    return [tool(getattr(ToolRegistry, name)) for name in names]


def _build_tools(skills: dict) -> list:
    """Return the run's toolset: the standard tools, plus ``use_skill`` iff skills
    were discovered. With no skills the result is byte-identical to today (D5).
    """
    tools = _all_tools()
    if skills:
        tools.append(tool(ToolRegistry.use_skill))
    return tools


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
    show_default="prompts interactively in the terminal",
    metavar="TEXT",
    help="Task for the agent. Without this, the agent prompts interactively.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    show_default="none — run a normal agent session instead",
    type=click.Path(),
    metavar="PATH",
    help=(
        "Path to a declarative YAML run config file (LLM connection, task, "
        "inline checks — see examples/*.yaml). Runs non-interactively through "
        "the eval-run engine and exits 0 on pass, 1 on fail, 2 on invalid config."
    ),
)
@click.option(
    "--max-steps",
    default=DEFAULT_MAX_STEPS,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Maximum agent loop steps.",
)
@click.option(
    "--resume",
    "resume_id",
    default=None,
    show_default="none",
    metavar="SESSION_ID",
    help="Resume a dead session from its last checkpoint (new linked session).",
)
@click.option(
    "--resume-last",
    is_flag=True,
    default=False,
    show_default=True,
    help="Resume the most recently checkpointed session.",
)
@click.option(
    "--no-safety-gate",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Disable the dangerous-command refusal gate for this run "
        "(same effect as MCA_DISABLE_DANGEROUS_COMMAND_GATE=1). See SECURITY.md."
    ),
)
@click.option(
    "--no-egress-filter",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Disable the network egress filter for this run (same effect as "
        "MCA_DISABLE_EGRESS_FILTER=1). See SECURITY.md."
    ),
)
@click.option(
    "--no-exfil-guard",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Disable the secret-exfiltration guard for this run "
        "(same effect as MCA_DISABLE_EXFIL_GUARD=1). See SECURITY.md."
    ),
)
@click.option(
    "--sandbox",
    "use_sandbox",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Run every bash subprocess inside an OS-level sandbox (macOS Seatbelt): "
        "writes confined to the workspace + temp allowlist, network denied by "
        "default. Refuses to run bash unsandboxed if unsupported on this host."
    ),
)
@click.version_option(version=__version__, prog_name="my-coding-agent")
def main(
    prompt: str | None,
    config_path: str | None,
    max_steps: int,
    resume_id: str | None,
    resume_last: bool,
    no_safety_gate: bool,
    no_egress_filter: bool,
    no_exfil_guard: bool,
    use_sandbox: bool,
) -> None:
    """Run the coding-agent pipeline.

    With no arguments this always starts an interactive terminal session
    (multi-line prompt, history, Esc-Enter to submit) — see \b--prompt to
    skip it and \b--config to run a declarative YAML config instead.

    \b
    Examples:
      uv run my-coding-agent
      uv run my-coding-agent -p "write tests for llm.py"
      uv run my-coding-agent --config examples/eval_run_config.yaml
      uv run my-coding-agent --resume 3f9a1c2b4d5e
      uv run my-coding-agent --resume-last
    """
    if no_safety_gate:
        os.environ[policy.DISABLE_ENV_VAR] = "1"
        click.secho(
            "⚠ --no-safety-gate: the dangerous-command refusal gate is OFF for "
            "this run. Every bash command the model emits executes verbatim, "
            "with no pre-execution check. See SECURITY.md.",
            fg="red",
            bold=True,
            err=True,
        )
    if no_exfil_guard:
        os.environ[exfil.DISABLE_ENV_VAR] = "1"
        click.secho(
            "⚠ --no-exfil-guard: the secret-exfiltration guard is OFF for "
            "this run. Outbound requests are never checked for known-sensitive "
            "content, with no pre-send check. See SECURITY.md.",
            fg="red",
            bold=True,
            err=True,
        )

    if no_egress_filter:
        os.environ[egress.schema.DISABLE_ENV_VAR] = "1"
        click.secho(
            "⚠ --no-egress-filter: the network egress filter is OFF for this "
            "run. `fetch_web` destinations are not checked against the "
            "known-malicious blocklist. See SECURITY.md.",
            fg="red",
            bold=True,
            err=True,
        )

    if use_sandbox:
        os.environ[sandbox.ENV_VAR] = "1"
        capability = sandbox.probe_host_capability()
        if not capability.supported:
            click.secho(
                f"✗ --sandbox: unavailable on this host ({capability.reason}). "
                "Refusing to run bash unsandboxed — every bash call this run "
                "will fail rather than execute unconfined.",
                fg="red",
                bold=True,
                err=True,
            )
        else:
            click.secho(
                "🔒 --sandbox: bash subprocesses run inside an OS-level sandbox "
                "for this run (writes confined to the workspace + temp "
                "allowlist, network denied by default).",
                fg="cyan",
                err=True,
            )

    if config_path:
        if resume_id or resume_last:
            raise click.UsageError(
                "--config cannot be combined with --resume/--resume-last."
            )
        _run_from_config(config_path)
        return

    if resume_id or resume_last:
        agent = _build_resumed_agent(resume_id, resume_last)
    else:
        agent = _build_fresh_agent(prompt)

    agent.execute(max_steps=max_steps)
    _exit_on_failure(agent)


def _run_from_config(config_path: str) -> None:
    """Load, validate, and execute a declarative YAML run config, then exit.

    Mirrors the (now removed) ``my-coding-agent-eval run --config`` subcommand:
    a validation failure exits 2 with every problem named, before any agent
    run or session directory is created; a scored run exits 0 on pass, 1 on
    fail.
    """
    path = Path(config_path)
    try:
        result, verdict = execute_from_config(path)
    except ConfigValidationError as exc:
        click.secho(f"Invalid config {path}:", fg="red", err=True)
        for problem in exc.problems:
            click.echo(f"  - {problem}", err=True)
        sys.exit(2)

    click.echo(f"Run {result.run_id}: verdict {verdict}")
    render_verdict(result)
    click.echo(f"Result written to {RESULTS_ROOT / result.run_id}/result.json")
    sys.exit(0 if verdict == "pass" else 1)


def _build_fresh_agent(prompt: str | None) -> AgentNode:
    """Construct a new run's agent from --prompt, or the interactive UI by default."""
    if prompt:
        user_prompt = prompt
    else:
        user_prompt = _read_interactive_prompt()
        if not user_prompt:
            click.secho("No prompt entered — using default.", fg="yellow", err=True)
            user_prompt = _DEFAULT_PROMPT

    # Discover skills once at session start (before the first LLM call). The
    # index is placed into the opening user message by AgentNode; the system
    # prompt is untouched so the #75 prefix-cache invariant holds.
    skills = discover_skills()
    return AgentNode(
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        tools=_build_tools(skills),
        label="Main Agent",
        skills=skills,
    )


def _build_resumed_agent(resume_id: str | None, resume_last: bool) -> AgentNode:
    """Load a checkpoint and build a fresh linked agent, or refuse and exit (D5).

    Refuses cleanly (exit 2, touching nothing) when no session id is given, none
    is resumable, or the checkpoint is missing/unreadable.
    """
    session_id = resume_id
    if resume_last:
        session_id = find_last_resumable(_SESSIONS_DIR)
        if session_id is None:
            click.secho(
                f"No resumable session with a checkpoint found under {_SESSIONS_DIR}/.",
                fg="red",
                err=True,
            )
            sys.exit(2)
    assert session_id is not None  # one of the two branches set it
    try:
        checkpoint = load_checkpoint(_SESSIONS_DIR / session_id)
    except CheckpointError as exc:
        click.secho(f"Cannot resume: {exc}", fg="red", err=True)
        sys.exit(2)
    click.secho(
        f"Resuming session {session_id} from step {checkpoint.step_num} "
        f"({len(checkpoint.messages)} messages).",
        fg="cyan",
        err=True,
    )
    # The checkpointed conversation already carries any skill index in its
    # messages, so ``from_checkpoint`` does not re-place it; but the discovered
    # skills still populate the registry so a resumed run can service
    # ``use_skill`` and load skill bodies on demand.
    skills = discover_skills()
    return AgentNode.from_checkpoint(
        checkpoint,
        tools=_build_tools(skills),
        skills=skills,
        label="Main Agent (resumed)",
    )


def _exit_on_failure(agent: AgentNode) -> None:
    """Exit non-zero with a one-line resume hint when the run failed (D6)."""
    if agent.failure_error is None:
        return
    resume_id = agent.failure_session_id or agent.session_id
    click.secho(
        f"Run stopped: unrecoverable LLM failure "
        f"({agent.failure_error.classification}). "
        f"Resume with: uv run my-coding-agent --resume {resume_id}",
        fg="red",
        err=True,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

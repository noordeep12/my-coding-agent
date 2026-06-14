# my-coding-agent

A personal, from-scratch Python agent framework for running LLM-powered coding agents against local LLM servers. Built for me.

## What It Is

A self-contained agentic loop — no external agent frameworks. The agent calls a local model, executes tool calls, feeds results back, and repeats until the task is done or a step/context limit is hit.

Key ideas:
- **Local-first**: targets OpenAI-compatible endpoints (MLX Server, Ollama) running on your machine
- **Decorator-based tools**: plain Python functions become LLM-callable tools
- **Context handoff**: when the context window fills up, the agent writes a structured summary of progress and spawns a fresh continuation — so long-running tasks don't get silently truncated
- **Main agent by default, optional stages**: a bare run is the Main Agent alone; an opt-in **Discovery** agent (`--discover`) can map the codebase first, and an opt-in **Session Analyzer** (`--analyze`) can review the run afterward
- **Session persistence**: each run saves token usage, tool calls, and a final summary to `.my_coding_agent/<session_id>/`

## Quick Start

```bash
# Start a local LLM server
mlx-llm server --model Qwen3.6-35B-A3B-4bit
# or
ollama serve

# Install dependencies and run
uv sync
uv run my-coding-agent
```

## Installing & Updating Locally

This is a [src-layout](https://hatch.pypa.io/latest/config/build/#packages) package managed with `uv`.

```bash
# Install runtime deps into a local .venv (Python 3.12+ is required)
uv sync

# Include the dev tools (ruff, mypy, pytest, sphinx, ...)
uv sync --all-groups
```

`uv run my-coding-agent` runs the package straight from `src/`, so **any code change is picked up on the next run with no reinstall step** — there is no separate `pip install -e .` to repeat. Re-run `uv sync` only when dependencies in `pyproject.toml` change.

## Entrypoint

The `my-coding-agent` console script is declared in `pyproject.toml`:

```toml
[project.scripts]
my-coding-agent = "my_coding_agent.workflows.main:main"
```

It points to `main()` in `src/my_coding_agent/workflows/main.py`. To change the default command, edit that mapping (and run `uv sync` to refresh the script); to change the default behaviour, edit the Click options on `main()`.

## CLI Usage

```bash
# Default: Main Agent only
uv run my-coding-agent

# Custom prompt:
uv run my-coding-agent --prompt "Your task here"

# Interactive paste mode (Esc then Enter, or Meta/Alt+Enter to submit; Ctrl+C to cancel)
uv run my-coding-agent --interactive

# Map the codebase first with the Discovery Agent (opt-in)
uv run my-coding-agent --discover

# Also run the Session Analyzer after the Main Agent completes (opt-in)
uv run my-coding-agent --analyze

# Help
uv run my-coding-agent --help
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--prompt`, `-p` | default commit-and-push task | Task for the Main Agent |
| `--interactive`, `-i` | `False` | Read the task prompt interactively from stdin (paste freely; Esc then Enter, or Meta/Alt+Enter to submit; Ctrl+C to cancel) |
| `--discover` / `--no-discover` | `--no-discover` (disabled) | Run the Discovery Agent before the Main Agent |
| `--max-steps` | `20` | Maximum agent loop steps for the Main Agent |
| `--analyze` / `--no-analyze`, `-a` / `-A` | `--no-analyze` (disabled) | Run the Session Analyzer Agent after the Main Agent completes |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |


## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Running local LLM server (MLX Server, Ollama, etc.)

## Architecture & Contributing

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the module layout, agent loop, tool dispatch, context handoff, and session persistence.
- [`CONTRIBUTE.md`](CONTRIBUTE.md) — development standards, tooling, testing, and commit conventions.

## Documentation

Build the API reference locally:

```bash
uv run sphinx-build -W -b html docs docs/_build/html
```

# my-coding-agent

A personal, from-scratch Python agent framework for running LLM-powered coding agents against local LLM servers. Built for me.

## What It Is

A self-contained agentic loop — no external agent frameworks. The agent calls a local model, executes tool calls, feeds results back, and repeats until the task is done or a step/context limit is hit.

Key ideas:
- **Local-first**: targets OpenAI-compatible endpoints (MLX Server, Ollama) running on your machine
- **Node-based pipeline**: the agentic loop is a DAG of named nodes (`ContextGuard → ToolRouting → LLMCall → ToolDispatch → FinalizeStep`) with an explicit data contract (`RunContext`) flowing between them
- **Decorator-based tools**: plain Python functions become LLM-callable tools
- **Context handoff**: when the context window fills up, the agent writes a structured summary of progress and spawns a fresh continuation — so long-running tasks don't get silently truncated
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
my-coding-agent = "my_coding_agent.pipeline.examples.simple:main"
```

It points to `main()` in `src/my_coding_agent/pipeline/examples/simple.py`. To change the default command, edit that mapping (and run `uv sync` to refresh the script); to change the default behaviour, edit the Click options on `main()`.

## CLI Usage

```bash
# Default: run with the default commit-and-push task
uv run my-coding-agent

# Custom prompt:
uv run my-coding-agent --prompt "Your task here"

# Interactive paste mode (Esc then Enter, or Meta/Alt+Enter to submit; Ctrl+C to cancel)
uv run my-coding-agent --interactive

# Help
uv run my-coding-agent --help
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--prompt`, `-p` | default commit-and-push task | Task for the agent |
| `--interactive`, `-i` | `False` | Read the task prompt interactively (Esc then Enter, or Meta/Alt+Enter to submit; Ctrl+C to cancel) |
| `--max-steps` | `50` | Maximum pipeline step iterations |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |


## Observability

Every run automatically records a structured `events.jsonl` alongside the other session files under `.my_coding_agent/<session_id>/`. No configuration needed — capture is always on.

| File | Contents |
|---|---|
| `stderr.log` | Plain-text log of the full run |
| `session_data.json` | Metrics, tool records, LLM call log, stop reason, usage rollup (own + delegated subagents, per call kind) |
| `events.jsonl` | Structured event stream (LLM calls, tool I/O, handoffs) |
| `artifacts/<tool_call_id>.<stream>.txt` | Full content of each offloaded large output stream (`stdout`/`stderr`), written at creation so bash can skim it during the run |
| `tool_artifacts.json` | End-of-run audit dump of the in-memory artifact records |

### Trace Explorer

Visualise sessions in a browser with an interactive pipeline DAG:

```bash
my-coding-agent-traces          # defaults: port 7474, dir .my_coding_agent
my-coding-agent-traces --port 8080 --dir /path/to/.my_coding_agent
```

Then open `http://localhost:7474`. The UI (an Apple-minimalist Preact app, served fully offline) shows:

- **Left pane** — a nested **Tree** of the run: the Main Agent's pipeline, with each delegated **subagent** nested (collapsible, with a coloured rail and badge) where it was spawned; each node label summarises what it added to the context window (e.g. *+196 assistant*, *+1,501 tool*)
  - Navigate with **↑/↓ (or j/k)** arrow keys; the focused node auto-selects
  - **Filters** button — show/hide nodes by type
- **Right pane** — node detail:
  - a single **Context window** box: the running window (total / max · %) and a composition bar + legend split into **system / user / assistant / tool** tokens, so you can see which role is inflating the window — subagent nodes show that subagent's **own** window, badged with its id
  - every node has the same three collapsible sections — **Outputs**, **Inputs**, **Attributes**
  - in **Outputs**, **Tool Dispatch** nodes render a status badge (**✓ success** / **✗ error**), the command that ran, and the output/error; **LLM Call** nodes render the response text, reasoning, and tool calls — each in a mini-editor
  - in **Inputs**, **LLM Call** nodes show both the `messages` sent to the model and the `tools` definitions the model was given that turn (the tool schemas that reached the LLM, not just the calls it suggested)
  - every content box (JSON and raw text) is a **mini VS Code-style editor** (powered by CodeMirror, vendored offline): syntax highlighting, line numbers, and code folding; a clickable **schema breadcrumb** for JSON (e.g. `root › [0] › function`); **collapse all / expand all**, **copy all**, and **find** with keyboard next/previous (Enter / Shift+Enter)
- The current **session id** is shown in the header (click to copy); the picker dropdown lists all sessions
- Loop-detected tool calls are flagged inline
- A **Breakdown** toggle in the stats bar (shown whenever the session has token-usage data) reveals per-call-kind and per-agent token totals across the whole delegated tree

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

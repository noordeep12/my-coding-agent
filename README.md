# my-coding-agent

A minimal Python agent library (~500 lines) that connects to local OpenAI-compatible LLM servers (MLX Server, Ollama) and supports decorator-based tool calling.

## Features

- **Local LLM Integration**: Connect to MLX Server, Ollama, or any OpenAI-compatible API
- **Tool Calling**: Decorator-based tool registry with built-in `bash`, `read_file`, `write_file`, and `read_article` tools
- **Multi-Step Agent Loop**: Automatic reasoning with tool execution and result integration
- **Context Window Handoff**: Automatic context reset when tokens approach the limit, with structured state transfer to a continuation agent
- **Discovery Agent**: Explores the workspace and writes a stable `discovery.md` for context continuity across sessions
- **Workflow CLI**: Click-based CLI orchestrating the full pipeline: Discovery → Main Agent
- **Colored Logging**: Custom log levels (TOOL, API, LLM) with ASCII startup banner and run summary

## Quick Start

```bash
# Start a local LLM server
mlx-llm server --model qwen3:35b
# or
ollama serve

# Install dependencies and run
uv sync
uv run python workflows/main.py
```

## CLI Usage

```bash
# Full pipeline (Discovery + Main Agent, discovery enabled by default)
uv run python workflows/main.py

# Custom prompt
uv run python workflows/main.py --prompt "Your task here"

# Interactive paste mode (Ctrl+D to end)
uv run python workflows/main.py --interactive

# Skip discovery stage
uv run python workflows/main.py --no-discover

# Help
uv run python workflows/main.py --help
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--prompt`, `-p` | default commit-and-push task | Task for the Main Agent |
| `--interactive`, `-i` | `False` | Read the task prompt interactively from stdin (paste freely, Ctrl+D to submit) |
| `--discover` / `--no-discover` | `--discover` (enabled) | Run the Discovery Agent before the Main Agent |
| `--max-steps` | `20` | Maximum agent loop steps for the Main Agent |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |

## Architecture

```
src/my_coding_agent/
├── __init__.py     — Re-exports: LLM, Agent, tool, ToolsRegistry, ContextHandoff
├── llm.py          — LLM base class: HTTP session, model discovery, chat_completion
├── agent.py        — Agent class: multi-step reasoning loop, context-window handoff
├── tools.py        — ToolsRegistry: bash, read_file, write_file, read_article
├── handoff.py      — ContextHandoff dataclass: structured state transfer between context windows
├── logger.py       — Colored logging (TOOL/API/LLM levels), startup banner, run summary
└── utils.py        — Response parsers: extract_message, extract_finish_reason, extract_usage, parse_tool_args

agents/
└── agentic_discovery.py  — Discovery Agent: explores workspace, writes discovery.md

workflows/
└── main.py              — Full workflow CLI: Discovery → Main Agent
```

## Available Tools

| Tool | Signature | Description |
|---|---|---|
| `bash` | `bash(command: str) -> str` | Run a shell command and return stdout, stderr, and exit code. |
| `read_file` | `read_file(path: str) -> str` | Read and return the full contents of a file at the given path. |
| `write_file` | `write_file(path: str, content: str) -> str` | Write content to a file, creating parent directories if needed. |
| `read_article` | `read_article(url: str) -> str` | Fetch a web page and return its content as clean markdown (max ~6000 tokens). |

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Running local LLM server (MLX Server, Ollama, etc.)

## Dependencies

- `click>=8.1` — CLI framework for workflow orchestration
- `colorama>=0.4.6` — Colored terminal output
- `html2text>=2025.4.15` — Web-to-markdown conversion (read_article tool)
- `httpx>=0.28.1` — HTTP client for LLM API calls
- `pydantic>=2.13.4` — Data validation
- `rich>=15.0.0` — Rich markdown rendering in run summaries

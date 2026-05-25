# my-coding-agent

A minimal Python agent library (~500 lines) that connects to local OpenAI-compatible LLM servers and supports tool calling.

## Features

- **Local LLM Integration**: Connect to MLX Server, Ollama, or any OpenAI-compatible API
- **Tool Calling**: Decorator-based tool registry with built-in `bash`, `read_file`, and `write_file` tools
- **Multi-Step Agent Loop**: Automatic reasoning with tool execution and result integration
- **Colored Logging**: Custom log levels (TOOL, API, LLM) with ASCII startup banner
- **Git Context**: Automatic repository detection and branch information

## Quick Start

```bash
# Start a local LLM server
mlx-llm server --model qwen3:35b
# or
ollama serve

# Run the agent
uv sync
uv run python examples/agentic_shell.py
```

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
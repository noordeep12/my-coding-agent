# my-coding-agent

A minimal Python agent that connects to a local OpenAI-compatible LLM (e.g. MLX, Ollama) and supports tool calling via a simple decorator-based registry.

## Requirements

- A running local LLM server at `http://127.0.0.1:8321/v1` (e.g. [MLX Server](https://github.com/ml-explore/mlx-examples))
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python agent.py "What is the weather in Tokyo?"
```

## Adding tools

Add methods to `ToolsRegistry` in `agent.py` — the `@tool` decorator auto-generates the JSON schema for the LLM.

```python
class ToolsRegistry:
    def get_weather(self, location):
        """Get the current weather for a location."""
        return f"Sunny in {location}."
```

## Configuration

Set the following constants at the top of `agent.py`:

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |

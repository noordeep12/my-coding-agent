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

---

## Agentic Shell Demo

A full agentic workflow demo that demonstrates the agent's ability to interact with the filesystem and execute shell commands autonomously.

### What it does

The agentic shell demo (`examples/agentic_shell.py`) shows the agent performing multi-step tasks using:

- **`bash(command)`** — Executes shell commands and returns stdout, stderr, and exit code
- **`read_file(path)`** — Reads and returns the full contents of a file
- **`write_file(path, content)`** — Writes content to a file, creating parent directories if needed

The agent uses these tools to complete complex tasks like:
- Reading and modifying files (e.g., updating documentation)
- Running git operations (commit, push)
- Querying external data sources
- Performing multi-step workflows that require decision-making between steps

### How to run it

1. **Start a local LLM server** (e.g., using MLX Server):
   ```bash
   # Example using MLX Server
   mlx-llm server --model qwen3:35b
   ```

2. **Run the agentic shell demo**:
   ```bash
   uv run python examples/agentic_shell.py
   ```

3. **Customize the task**: Edit the `messages` list in `examples/agentic_shell.py` to define your own prompt:
   ```python
   messages = [
       {
           "role": "system",
           "content": (
               "You are a helpful assistant. Use tools when needed. "
               "Use absolute paths when working with files."
               # ... additional context ...
           )
       },
       {
           "role": "user",
           "content": "Your task here..."
       }
   ]
   ```

4. **Adjust the maximum steps** (default: 20) to control how many reasoning loops the agent performs:
   ```python
   agent = Agent(messages=messages, tools=tools)
   final_messages = agent.run(max_steps=20)  # Increase for complex tasks
   ```

### Example tasks

- Update project documentation based on code changes
- Perform git operations with standardized commit messages
- Collect and process external data (e.g., vulnerability databases)
- Navigate and modify file structures autonomously

### Output

The agent prints the full conversation history, including all tool calls and their results:

```
Initial messages:  [...]
Available tools:   [...]

[step 1] tokens — prompt: 512, completion: 128, total: 640
[step 2] tokens — prompt: 800, completion: 256, total: 1056
...

Final messages:  [...]
```

This allows you to trace exactly how the agent reasoned through each step and what tools it used.

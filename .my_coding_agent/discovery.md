# Discovery Summary — my-coding-agent

> **Generated:** 2025-05-25  
> **Branch:** `main` (clean working tree)  
> **Workspace:** `/Users/noordeepsingh/Workspace/my-coding-agent`  
> **Python:** 3.12 (via `uv`)

---

## 1. Codebase Overview

**my-coding-agent** is a minimal Python agent library (~500 lines) that connects to local OpenAI-compatible LLM servers (MLX Server, Ollama) and supports decorator-based tool calling.

### Architecture

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

examples/
└── sample.ipynb         — Jupyter notebook example
```

### Key Components

| Module | Purpose |
|---|---|
| `LLM` | Base class wrapping httpx sessions to OpenAI-compatible APIs. Discovers available models and context windows. |
| `Agent` | Extends LLM with a multi-step agent loop. Supports automatic context-window handoff when tokens approach the limit. |
| `ToolsRegistry` | Static method registry for tools (`bash`, `read_file`, `write_file`, `read_article`). |
| `ContextHandoff` | Dataclass that serializes agent state (step, tokens, content) to `.my_coding_agent/handoffs/` for context reset. |
| `log_parser` | (Removed) Session log parser was removed in recent refactor. |
| `agentic_discovery` | Agent that explores the workspace and writes a stable `discovery.md` context document. |
| `workflows/main.py` | Click-based CLI orchestrating the full pipeline: Discovery → Main Agent. |

### Dependencies

- Python 3.12+
- `click>=8.1` — CLI framework for workflow orchestration
- `colorama>=0.4.6` — Colored terminal output
- `html2text>=2025.4.15` — Web-to-markdown conversion (read_article tool)
- `httpx>=0.28.1` — HTTP client for LLM API calls
- `ipdb>=0.13.13` — Interactive debugger
- `pydantic>=2.13.4` — Data validation
- `rich>=15.0.0` — Rich markdown rendering in run summaries
- `uv` — Dependency/project management

### Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |

---

## 2. Git Repository Status

### Current State

- **Working tree:** Clean (no uncommitted changes)
- **On branch:** `main`
- **Up to date with:** `origin/main`

### Recent Commit History (last 10)

| Hash | Subject |
|---|---|
| `de46a70` | examples |
| `787a753` | refactor: remove session analyzer, add read_article tool, fix token ratio |
| `d716275` | feat(logger): enhance startup banner with agent title, metrics, and tool listing |
| `78d36e9` | feat(workflows): enable discover/analyze by default, show all defaults in --help |
| `faf2859` | fix(workflows): fix ModuleNotFoundError for agents package |
| `62290d5` | refactor: rename examples/ to agents/ |
| `33a4cc5` | refactor(workflows): introduce workflows/ with Click-based CLI |
| `2c9d71f` | refactor(examples): extract Discovery Agent into agentic_discovery.py |
| `1fbcb47` | feat(shell): add --discover and --analyze flags to agentic_shell |
| `d3f2dec` | fix(shell): clean up demo code and improve discovery agent output |

### Notable Features in Recent History

- **Context Reset with Handoff:** The agent automatically detects when its context window is near capacity (configurable threshold, default 75%), generates a structured handoff summary, and spawns a continuation agent with a fresh context.
- **read_article Tool:** Fetches web pages and converts them to clean markdown (truncated at ~6000 tokens to prevent context explosion).
- **CLI Argument Parsing:** Supports `--prompt` / `-p` for custom user messages and `--interactive` / `-i` for paste-mode input.
- **Workflow Pipeline:** Full pipeline with Discovery → Main Agent, configurable via Click CLI.
- **Enhanced Logging:** Startup banner and run summary with agent labels, message counts, context reset thresholds, tool listings, and handoff event rendering.

---

## 3. Insights & Observations

### Design Patterns

1. **Decorator-based Tool Registration:** The `@tool` decorator (actually `function_to_json`) converts Python functions into OpenAI-compatible tool definition dicts, enabling LLMs to discover and invoke tools dynamically.

2. **Context Window Protection:** The agent monitors token usage per step (via API-reported `prompt_tokens`) and triggers a context handoff before exceeding the model's context limit. The handoff is persisted to `.my_coding_agent/handoffs/` for auditability.

3. **Structured Logging:** Custom log levels (TOOL=15, API=25, LLM=35) with color-coded output routed to stderr. The startup banner and run summary provide rich, formatted information about the agent session.

4. **Pipeline Architecture:** The workflow system chains agents together — Discovery Agent maps the workspace, Main Agent executes tasks.

### Architecture Decisions

- **No framework dependencies:** The agent is intentionally minimal (~500 lines) with no heavy framework dependencies beyond httpx, pydantic, and rich.
- **Local-first:** Designed for local LLM servers (MLX Server, Ollama) with sensible defaults.
- **Extensible tool registry:** New tools can be added as static methods on `ToolsRegistry` and registered via the `@tool` decorator.
- **Click-based CLI:** Workflow orchestration uses Click for robust argument parsing, help text, and discoverability.
- **Session Analyzer removed:** The session analyzer agent was removed in commit `787a753` to simplify the codebase and focus on core functionality.

### Uncommitted Changes

None — the working tree is clean and up to date with origin/main.

---

## 4. Tool Descriptions

### Available Tools (ToolsRegistry)

| Tool | Signature | Description |
|---|---|---|
| `bash` | `bash(command: str) -> str` | Run a shell command and return stdout, stderr, and exit code. Use for running tests, installing packages, git operations, or any shell task. |
| `read_file` | `read_file(path: str) -> str` | Read and return the full contents of a file at the given path. Use to inspect source code, configs, or any text file before editing. |
| `write_file` | `write_file(path: str, content: str) -> str` | Write content to a file, creating parent directories if needed. Use to create new files or overwrite existing ones. |
| `read_article` | `read_article(url: str) -> str` | Fetch a web page and return its content as clean markdown (max ~6000 tokens). Use when the user provides a URL or link to an article, blog post, or documentation page. |

### How to Register New Tools

```python
from my_coding_agent import ToolsRegistry, tool

class ToolsRegistry:
    @staticmethod
    def my_custom_tool(arg1: str, arg2: int) -> str:
        """Description of what the tool does."""
        # implementation
        return result

# Register with agent
tools = [tool(ToolsRegistry.my_custom_tool)]
agent = Agent(tools=tools)
```

The `@tool` decorator (actually `function_to_json`) automatically converts Python functions into OpenAI-compatible tool definition dicts based on the function's signature and docstring.

### How to Use the Workflow CLI

```bash
# Full pipeline (Discovery + Main Agent)
uv run python workflows/main.py

# Custom prompt
uv run python workflows/main.py --prompt "Your task here"

# Interactive paste mode (Ctrl+D to end)
uv run python workflows/main.py --interactive

# Skip discovery
uv run python workflows/main.py --no-discover
```

---

## 5. Latest User Requests

Based on the git history and current workspace, the most recent user-facing requests that have been addressed:

1. **Context Reset with Handoff:** Implement automatic context window management with structured state transfer between context windows when the agent approaches its token limit.

2. **Web Content Fetching:** Add a `read_article` tool that fetches web pages and converts them to clean markdown for agent consumption.

3. **Custom Prompts via CLI:** Add CLI argument parsing (`--prompt`, `--interactive`) to allow users to pass custom prompts to the agent from the command line.

4. **Discovery Agent Workflow:** Implement an automated discovery step that explores the workspace and creates a `discovery.md` file to provide context continuity across sessions.

5. **Workflow Pipeline:** Introduce a Click-based CLI that chains Discovery → Main Agent, with configurable options for each stage.

6. **Enhanced Logging:** Improve the startup banner and run summary with agent labels, message counts, context reset thresholds, tool listings, and handoff event rendering.

7. **Simplification:** Remove the session analyzer and examples directory to focus on core functionality (most recent commits).

---

## 6. Recent Conversation Transcript

### Session: Discovery Agent (Current)

**User Request:**
> Explore the workspace and discover any relevant information that can help you understand the current state of the codebase, git repository, recent activities, and anything else that might be useful for future tasks. Summarise your findings into a Markdown file saved to `.my_coding_agent/discovery.md` in the current workspace. It should contain:
> 1. A summary of the current state of the codebase and git repository
> 2. Any recent activities or changes that might be relevant
> 3. Insights or observations that help understand the context better
> 4. Tool descriptions and how to use them
> 5. Latest user requests
> 6. A recent transcript of the conversation
>
> The goal is to create a stable prompt prefix that can be prepended to future conversations to provide context and continuity, even if the conversation history gets truncated due to token limits.

**Tools Used:**
- `bash(command)` — Explored workspace structure (`find`, `git log`, `git status`, `git diff --stat`, `ls -la`)
- `read_file(path)` — Read source files (`pyproject.toml`, `README.md`, `agent.py`, `tools.py`, `llm.py`, `handoff.py`, `logger.py`, `utils.py`, `agentic_discovery.py`, `workflows/main.py`)

**Key Findings:**
- The workspace contains a minimal Python agent library (~500 lines) connecting to local OpenAI-compatible LLM servers.
- The git repository has 20+ commits with features including context reset with handoff, web content fetching, CLI argument parsing, and discovery agent workflow.
- The working tree is clean and up to date with origin/main (6 commits ahead of origin/main have been committed).
- The agent supports automatic context window management with structured state transfer between context windows.
- The session analyzer was removed in a recent refactor to simplify the codebase.

---

## Quick Reference

### Running the Agent

```bash
# Start a local LLM server (MLX Server or Ollama)
mlx-llm server --model qwen3:35b
# or
ollama serve

# Run the agent
uv sync
uv run python workflows/main.py
```

### CLI Options

```bash
# Full pipeline (default, discovery enabled)
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

### Key File Paths

| Path | Purpose |
|---|---|
| `src/my_coding_agent/agent.py` | Main Agent class with multi-step loop and context handoff |
| `src/my_coding_agent/tools.py` | ToolsRegistry with bash, read_file, write_file, read_article |
| `src/my_coding_agent/handoff.py` | ContextHandoff for structured state transfer |
| `src/my_coding_agent/logger.py` | Colored logging, startup banner, run summary |
| `src/my_coding_agent/utils.py` | Response parsers for chat completion API |
| `agents/agentic_discovery.py` | Discovery Agent: explores workspace, writes discovery.md |
| `workflows/main.py` | Full pipeline CLI: Discovery → Main Agent |
| `.my_coding_agent/discovery.md` | This file — workspace context for future sessions |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |

### Git Status

- **Branch:** `main`
- **Status:** Clean working tree, up to date with origin/main

# Discovery Summary — my-coding-agent

> **Generated:** 2025-05-25  
> **Branch:** `main` (6 commits ahead of `origin/main`)  
> **Workspace:** `/Users/noordeepsingh/Workspace/my-coding-agent`  
> **Python:** 3.12+ (via `uv`)

---

## 1. Codebase Overview

**my-coding-agent** is a minimal Python agent library (~500 lines) that connects to local OpenAI-compatible LLM servers (MLX Server, Ollama) and supports decorator-based tool calling.

### Architecture

```
src/my_coding_agent/
├── __init__.py     — Re-exports: LLM, Agent, tool, ToolsRegistry, parse_session_log, ContextHandoff
├── llm.py          — LLM base class: HTTP session, model discovery, chat_completion
├── agent.py        — Agent class: multi-step reasoning loop, context-window handoff
├── tools.py        — ToolsRegistry: bash, read_file, write_file, read_article
├── handoff.py      — ContextHandoff dataclass: structured state transfer between context windows
├── logger.py       — Colored logging (TOOL/API/LLM levels), startup banner, run summary
├── utils.py        — Response parsers: extract_message, extract_finish_reason, extract_usage, parse_tool_args
└── log_parser.py   — Session log parser: parses ANSI-colored stderr logs into SessionLog dataclass

agents/
├── agentic_discovery.py  — Discovery Agent: explores workspace, writes discovery.md
└── agentic_session_analyzer.py — Session Analyzer: parses logs, writes analysis reports

workflows/
└── main.py              — Full workflow: Discovery → Main Agent → (optional) Session Analyzer
```

### Key Components

| Module | Purpose |
|---|---|
| `LLM` | Base class wrapping httpx sessions to OpenAI-compatible APIs. Discovers available models and context windows. |
| `Agent` | Extends LLM with a multi-step agent loop. Supports automatic context-window handoff when tokens approach the limit. |
| `ToolsRegistry` | Static method registry for tools (`bash`, `read_file`, `write_file`, `read_article`). |
| `ContextHandoff` | Dataclass that serializes agent state (step, tokens, content) to `.my_coding_agent/handoffs/` for context reset. |
| `log_parser` | Parses ANSI-colored session logs into structured `SessionLog` with per-step metrics and tool call records. |
| `agentic_discovery` | Agent that explores the workspace and writes a stable `discovery.md` context document. |
| `agentic_session_analyzer` | Agent that parses session logs, researches best practices, and writes structured analysis reports. |
| `workflows/main.py` | Click-based CLI orchestrating the full pipeline: Discovery → Main Agent → Session Analyzer. |

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

### Current State (uncommitted changes)

| File | Status |
|---|---|
| `.archive/agent.ipynb` | **Deleted** — archived notebook no longer needed |
| `.gitignore` | **Modified** — added `.my_coding_agent/handoffs/` and `.my_coding_agent/` to exclusions |
| `src/my_coding_agent/agent.py` | **Modified** — `print_banner()` now passes `label`, `n_messages`, and `context_reset_threshold` |
| `src/my_coding_agent/logger.py` | **Modified** — `print_banner()` enhanced with agent title, message count, reset threshold, and tool listing; `print_run_summary()` enhanced with handoff rendering and markdown support |

### Branch Info

- **On branch:** `main`
- **Ahead of `origin/main` by:** 6 commits (not yet pushed)
- **Unstaged changes:** 4 files changed, 62 insertions(+), 661 deletions(-)

### Recent Commit History (last 10)

| Hash | Subject |
|---|---|
| `78d36e9` | feat(workflows): enable discover/analyze by default, show all defaults in --help |
| `faf2859` | fix(workflows): fix ModuleNotFoundError for agents package |
| `62290d5` | refactor: rename examples/ to agents/ |
| `33a4cc5` | refactor(workflows): introduce workflows/ with Click-based CLI |
| `2c9d71f` | refactor(examples): extract Discovery Agent into agentic_discovery.py |
| `1fbcb47` | feat(shell): add --discover and --analyze flags to agentic_shell |
| `d3f2dec` | fix(shell): clean up demo code and improve discovery agent output |
| `5f04dfe` | feat(logger): show context reset events in run summary |
| `321dbff` | feat(agent): implement context reset with structured handoff (issue #8) |
| `dc26c80` | fix(harness): apply session analysis recommendations |

### Notable Features in Recent History

- **Context Reset with Handoff (issue #8):** The agent automatically detects when its context window is near capacity (configurable threshold, default 75%), generates a structured handoff summary, and spawns a continuation agent with a fresh context.
- **Session Analyzer:** An agent that parses stderr session logs into structured summaries for post-hoc analysis.
- **read_article Tool:** Fetches web pages and converts them to clean markdown (truncated at ~6000 tokens to prevent context explosion).
- **CLI Argument Parsing:** Supports `--prompt` / `-p` for custom user messages and `--interactive` / `-i` for paste-mode input.
- **Workflow Pipeline:** Full pipeline with Discovery → Main Agent → Session Analyzer, all configurable via Click CLI.

---

## 3. Insights & Observations

### Design Patterns

1. **Decorator-based Tool Registration:** The `@tool` decorator (actually `function_to_json`) converts Python functions into OpenAI-compatible tool definition dicts, enabling LLMs to discover and invoke tools dynamically.

2. **Context Window Protection:** The agent monitors token usage per step (via API-reported `prompt_tokens`) and triggers a context handoff before exceeding the model's context limit. The handoff is persisted to `.my_coding_agent/handoffs/` for auditability.

3. **Structured Logging:** Custom log levels (TOOL=15, API=25, LLM=35) with color-coded output routed to stderr. The startup banner and run summary provide rich, formatted information about the agent session.

4. **Session Log Parsing:** The `log_parser` module can parse ANSI-colored stderr logs into structured `SessionLog` objects, enabling automated analysis of agent sessions (token usage, tool calls, errors, timing).

5. **Pipeline Architecture:** The workflow system chains agents together — Discovery Agent maps the workspace, Main Agent executes tasks, and Session Analyzer provides post-hoc analysis.

### Architecture Decisions

- **No framework dependencies:** The agent is intentionally minimal (~500 lines) with no heavy framework dependencies beyond httpx, pydantic, and rich.
- **Local-first:** Designed for local LLM servers (MLX Server, Ollama) with sensible defaults.
- **Extensible tool registry:** New tools can be added as static methods on `ToolsRegistry` and registered via the `@tool` decorator.
- **Click-based CLI:** Workflow orchestration uses Click for robust argument parsing, help text, and discoverability.

### Uncommitted Changes (Pending Commit)

The current working tree has 4 modified/deleted files:
1. **`.archive/agent.ipynb`** — Deleted (archived notebook removed)
2. **`.gitignore`** — Updated to exclude `.my_coding_agent/handoffs/` and `.my_coding_agent/`
3. **`src/my_coding_agent/agent.py`** — `print_banner()` now passes `label`, `n_messages`, and `context_reset_threshold` parameters
4. **`src/my_coding_agent/logger.py`** — `print_banner()` enhanced with agent title row, message count, reset threshold, tool listing; `print_run_summary()` enhanced with handoff rendering and markdown support

These changes should be reviewed and committed before pushing to `origin/main`.

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
# Full pipeline (Discovery + Main + Analysis)
uv run python workflows/main.py

# Custom prompt
uv run python workflows/main.py --prompt "Your task here"

# Interactive paste mode (Ctrl+D to end)
uv run python workflows/main.py --interactive

# Skip discovery or analysis
uv run python workflows/main.py --no-discover --no-analyze
```

---

## 5. Latest User Requests

Based on the git history and current workspace, the most recent user-facing requests that have been addressed:

1. **Context Reset with Handoff (Issue #8):** Implement automatic context window management with structured state transfer between context windows when the agent approaches its token limit.

2. **Session Analysis:** Add a session analyzer agent that can parse and analyze stderr session logs to provide structured summaries of agent runs (token usage, tool calls, errors, timing).

3. **Web Content Fetching:** Add a `read_article` tool that fetches web pages and converts them to clean markdown for agent consumption.

4. **Custom Prompts via CLI:** Add CLI argument parsing (`--prompt`, `--interactive`) to allow users to pass custom prompts to the agent from the command line.

5. **Discovery Agent Workflow:** Implement an automated discovery step that explores the workspace and creates a `discovery.md` file to provide context continuity across sessions.

6. **Workflow Pipeline:** Introduce a Click-based CLI that chains Discovery → Main Agent → Session Analyzer, with configurable options for each stage.

7. **Enhanced Logging:** Improve the startup banner and run summary with agent labels, message counts, context reset thresholds, tool listings, and handoff event rendering.

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
- `read_file(path)` — Read source files (`pyproject.toml`, `README.md`, `agent.py`, `tools.py`, `llm.py`, `handoff.py`, `logger.py`, `utils.py`, `log_parser.py`, `agentic_discovery.py`, `agentic_session_analyzer.py`, `workflows/main.py`)

**Key Findings:**
- The workspace contains a minimal Python agent library (~500 lines) connecting to local OpenAI-compatible LLM servers.
- The git repository has 20+ commits with features including context reset with handoff, session analysis, web content fetching, CLI argument parsing, and discovery agent workflow.
- There are 6 uncommitted changes: deletion of `.archive/agent.ipynb`, modifications to `.gitignore`, `agent.py`, and `logger.py`.
- The branch is 6 commits ahead of `origin/main` and has not been pushed yet.
- The agent supports automatic context window management with structured state transfer between context windows.

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
# Full pipeline (default)
uv run python workflows/main.py

# Custom prompt
uv run python workflows/main.py --prompt "Your task here"

# Interactive paste mode (Ctrl+D to end)
uv run python workflows/main.py --interactive

# Skip discovery or analysis stages
uv run python workflows/main.py --no-discover --no-analyze

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
| `src/my_coding_agent/log_parser.py` | Session log parser for post-hoc analysis |
| `agents/agentic_discovery.py` | Discovery Agent: explores workspace, writes discovery.md |
| `agents/agentic_session_analyzer.py` | Session Analyzer: parses logs, writes reports |
| `workflows/main.py` | Full pipeline CLI: Discovery → Main → Analysis |
| `.my_coding_agent/discovery.md` | This file — workspace context for future sessions |
| `.my_coding_agent/handoffs/` | Directory for context handoff files (created during context resets) |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |

### Git Status

- **Branch:** `main`
- **Ahead of origin:** 6 commits (not yet pushed)
- **Unstaged changes:** 4 files (agent.py, logger.py, .gitignore, .archive/agent.ipynb)

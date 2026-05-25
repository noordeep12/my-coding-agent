# Discovery Summary — my-coding-agent

> **Generated:** 2025-05-25  
> **Branch:** `main` (up to date with `origin/main`)  
> **Workspace:** `/Users/noordeepsingh/Workspace/my-coding-agent`

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
```

### Key Components

| Module | Purpose |
|---|---|
| `LLM` | Base class wrapping httpx sessions to OpenAI-compatible APIs. Discovers available models and context windows. |
| `Agent` | Extends LLM with a multi-step agent loop. Supports automatic context-window handoff when tokens approach the limit. |
| `ToolsRegistry` | Static method registry for tools (`bash`, `read_file`, `write_file`, `read_article`). |
| `ContextHandoff` | Dataclass that serializes agent state (step, tokens, content) to `.my_coding_agent/handoffs/` for context reset. |
| `log_parser` | Parses ANSI-colored session logs into structured `SessionLog` with per-step metrics and tool call records. |

### Dependencies

- Python 3.12+
- `colorama` — Colored terminal output
- `html2text` — Web-to-markdown conversion (read_article tool)
- `httpx` — HTTP client for LLM API calls
- `ipdb` — Interactive debugger
- `pydantic` — Data validation
- `rich` — Rich markdown rendering in run summaries
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
| `.my_coding_agent/discovery.md` | **Deleted** (was previously generated) |
| `examples/agentic_shell.py` | **Modified** — Removed commented-out demo code, added `breakpoint()` for debugging, fixed tool name dynamic injection |

### Recent Commit History (last 15)

| Hash | Subject |
|---|---|
| `4867d94` | . |
| `321dbff` | feat(agent): implement context reset with structured handoff (issue #8) |
| `dc26c80` | fix(harness): apply session analysis recommendations |
| `2509354` | feat(examples): add agentic_session_analyzer agent for automated session analysis |
| `8b7fc25` | feat(tools): add read_article tool for fetching web content as markdown |
| `049351a` | feat(shell): add CLI argument parsing for custom prompts |
| `dda0051` | feat(agent): add context window protection to prevent exceeding limits |
| `8c846b1` | fix(llm): add fallback context window when model metadata unavailable |
| `bd1e4a4` | feat: add project README for GitHub |
| `0754574` | style: simplify run summary layout and align width with startup banner |
| `f33e7bb` | feat: enhance run summary with rich markdown rendering and wider display |
| `62ac339` | feat: add agent label and enhance run summary with tool call tracking and formatted output |
| `504c703` | feat: add run summary with tool call tracking and formatted output |
| `191bc06` | fix: route logging output to stderr instead of stdout |
| `628b71e` | feat: add discovery agent workflow for workspace exploration in agentic shell |

### Notable Features in Recent History

- **Context Reset with Handoff (issue #8):** The agent automatically detects when its context window is near capacity (configurable threshold, default 75%), generates a structured handoff summary, and spawns a continuation agent with a fresh context.
- **Session Analyzer:** An agent that parses stderr session logs into structured summaries for post-hoc analysis.
- **read_article Tool:** Fetches web pages and converts them to clean markdown (truncated at ~6000 tokens to prevent context explosion).
- **CLI Argument Parsing:** Supports `--prompt` / `-p` for custom user messages and `--interactive` / `-i` for paste-mode input.

---

## 3. Insights & Observations

### Design Patterns

1. **Decorator-based Tool Registration:** The `@tool` decorator (actually `function_to_json`) converts Python functions into OpenAI-compatible tool definition dicts, enabling LLMs to discover and invoke tools dynamically.

2. **Context Window Protection:** The agent monitors token usage per step (via API-reported `prompt_tokens`) and triggers a context handoff before exceeding the model's context limit. The handoff is persisted to `.my_coding_agent/handoffs/` for auditability.

3. **Structured Logging:** Custom log levels (TOOL=15, API=25, LLM=35) with color-coded output routed to stderr. The startup banner and run summary provide rich, formatted information about the agent session.

4. **Session Log Parsing:** The `log_parser` module can parse ANSI-colored stderr logs into structured `SessionLog` objects, enabling automated analysis of agent sessions (token usage, tool calls, errors, timing).

### Architecture Decisions

- **No framework dependencies:** The agent is intentionally minimal (~500 lines) with no heavy framework dependencies beyond httpx, pydantic, and rich.
- **Local-first:** Designed for local LLM servers (MLX Server, Ollama) with sensible defaults.
- **Extensible tool registry:** New tools can be added as static methods on `ToolsRegistry` and registered via the `@tool` decorator.

### Uncommitted Changes

The current working tree has:
- **Deleted** `.my_coding_agent/discovery.md` — This is the file we're creating now.
- **Modified** `examples/agentic_shell.py` — Cleaned up commented-out demo code, added a `breakpoint()` for debugging, and fixed dynamic tool name injection in the system prompt.

These changes should be reviewed and committed before proceeding with other work.

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

---

## 5. Latest User Requests

Based on the git history and current workspace, the most recent user-facing requests that have been addressed:

1. **Context Reset with Handoff (Issue #8):** Implement automatic context window management with structured state transfer between context windows when the agent approaches its token limit.

2. **Session Analysis:** Add a session analyzer agent that can parse and analyze stderr session logs to provide structured summaries of agent runs (token usage, tool calls, errors, timing).

3. **Web Content Fetching:** Add a `read_article` tool that fetches web pages and converts them to clean markdown for agent consumption.

4. **Custom Prompts via CLI:** Add CLI argument parsing (`--prompt`, `--interactive`) to allow users to pass custom prompts to the agent from the command line.

5. **Discovery Agent Workflow:** Implement an automated discovery step that explores the workspace and creates a `discovery.md` file to provide context continuity across sessions.

---

## 6. Recent Conversation Transcript

### Session: Discovery Agent Creation

**User Request:**
> Explore the workspace and discover any relevant information that can help you understand the current state of the codebase, git repository, recent activities, and anything else that might be useful for future tasks. Summarize your findings in a concise way into a Markdown file saved to `.my_coding_agent/discovery.md` in the current workspace. It should contains: 1. a summary of the current state of the codebase and git repository, 2. any recent activities or changes that might be relevant, 3. any insights or observations that can help you understand the context better. 4. Tool descriptions and how to use them. 5. Latest user requests. 6. Recent transcript of the conversation. The goal is to create a Stable prompt prefix that can be prepended to future conversations to provide context and continuity, even if the conversation history gets truncated due to token limits.

**Tools Used:**
- `bash(command)` — Explored workspace structure (`find`, `git log`, `git status`, `git diff --stat`, `ls -la`)
- `read_file(path)` — Read source files (`pyproject.toml`, `README.md`, `agent.py`, `tools.py`, `llm.py`, `handoff.py`, `logger.py`, `utils.py`, `log_parser.py`, `agentic_shell.py`)

**Key Findings:**
- The workspace contains a minimal Python agent library (~500 lines) connecting to local OpenAI-compatible LLM servers.
- The git repository has 15+ commits with features including context reset with handoff, session analysis, web content fetching, CLI argument parsing, and discovery agent workflow.
- The current working tree has uncommitted changes: deletion of `discovery.md` and modifications to `examples/agentic_shell.py`.
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
uv run python examples/agentic_shell.py
```

### CLI Options

```bash
# Custom prompt
uv run python examples/agentic_shell.py --prompt "Your task here"

# Interactive paste mode (Ctrl+D to end)
uv run python examples/agentic_shell.py --interactive
```

### Key File Paths

| Path | Purpose |
|---|---|
| `src/my_coding_agent/agent.py` | Main Agent class with multi-step loop and context handoff |
| `src/my_coding_agent/tools.py` | ToolsRegistry with bash, read_file, write_file, read_article |
| `src/my_coding_agent/handoff.py` | ContextHandoff for structured state transfer |
| `src/my_coding_agent/logger.py` | Colored logging, startup banner, run summary |
| `src/my_coding_agent/log_parser.py` | Session log parser for post-hoc analysis |
| `examples/agentic_shell.py` | Example runner with discovery agent workflow |
| `.my_coding_agent/discovery.md` | This file — workspace context for future sessions |
| `.my_coding_agent/handoffs/` | Directory for context handoff files (created during context resets) |

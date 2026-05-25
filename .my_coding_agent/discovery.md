# Discovery Report: my-coding-agent

**Generated:** 2025-05-25  
**Branch:** `main` (up to date with `origin/main`)  
**Remote:** `https://github.com/noordeep12/my-coding-agent.git`

---

## 1. Codebase Summary

**my-coding-agent** is a minimal Python agent library (v0.1.0) that connects to local OpenAI-compatible LLM servers (e.g., MLX Server, Ollama) and supports tool calling via a decorator-based registry.

### Project Structure

```
my-coding-agent/
├── src/my_coding_agent/
│   ├── __init__.py      # Package exports (LLM, Agent, tool, ToolsRegistry)
│   ├── llm.py           # LLM base class with HTTPX session, chat completion, tool execution
│   ├── agent.py         # Agent class with run loop, message management, step tracking
│   ├── tools.py         # ToolsRegistry with bash, read_file, write_file tools
│   ├── logger.py        # Colored logging with custom levels (TOOL, API, LLM) and ASCII banner
│   └── utils.py         # Response parsing helpers (extract_message, extract_usage, etc.)
├── examples/
│   └── agentic_shell.py # Agentic workflow demo with discovery + main agent
├── pyproject.toml       # Package config: Python >=3.12, deps: colorama, httpx, pydantic
├── uv.lock              # Dependency lock file
├── README.md            # Project documentation
├── sample.ipynb         # Jupyter notebook sample (142KB)
└── .archive/
    └── agent.ipynb      # Archived notebook (49KB)
```

### Key Components

| Module | Lines | Purpose |
|--------|-------|---------|
| `llm.py` | 130 | Base LLM class, HTTP session, chat completion, tool call execution |
| `agent.py` | 86 | Agent class with run loop, message management, max_steps control |
| `tools.py` | 97 | ToolsRegistry with `bash`, `read_file`, `write_file` tools |
| `logger.py` | 158 | Colored logging, custom levels, ASCII startup banner, git info |
| `utils.py` | 22 | Response parsing helpers |
| **Total** | **496** | **Core source code** |

### Configuration Defaults

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID to use |

### Requirements

- Python 3.12+
- Running local LLM server (MLX Server, Ollama, etc.)
- [uv](https://github.com/astral-sh/uv) for dependency management

---

## 2. Recent Git Activity

### Current State
- **Branch:** `main` (up to date with `origin/main`)
- **Unstaged changes:** `examples/agentic_shell.py` (modified but not committed)
- **Recent commit:** `628b71e` - "feat: add discovery agent workflow for workspace exploration in agentic shell"

### Recent Commits (Last 10)

| Hash | Type | Message |
|------|------|---------|
| `628b71e` | feat | add discovery agent workflow for workspace exploration in agentic shell |
| `8be8d57` | feat | add label parameter to Agent for improved identification and logging |
| `fe30f59` | feat | enhance system prompt with git context and improve tool formatting |
| `1038c82` | style | add trailing newlines to LLM log output for improved readability |
| `452a8eb` | style | move newline separators before LLM log content for improved readability |
| `42077cb` | style | add extra newlines to LLM log output for improved readability |
| `cc3a9fc` | refactor | remove call/result prefixes from LLM log messages for cleaner output |
| `17bc6af` | refactor | simplify log message format in agent.py |
| `7a4d811` | refactor | add labels to LLM log output for improved clarity |
| `d6eb3ad` | refactor | remove extra newlines from LLM log output formatting |

### Development Pattern
The project has gone through extensive **logging refinement** (refactoring log output format, adding labels, adjusting colors) and **feature enhancements** (discovery agent, label parameter, git context). The most recent work focuses on adding a discovery agent workflow.

---

## 3. Key Insights & Observations

### Architecture
- **Agent Loop:** The `Agent.run()` method implements a multi-step loop where:
  1. Messages are sent to the LLM
  2. Tool calls in the response are executed via `ToolsRegistry`
  3. Results are added back to the conversation
  4. Token usage is tracked with context window percentage
  5. Loop exits on `stop`/`exit`/`quit` finish reason or `max_steps`

### Tool System
- Tools are registered via the `@tool` decorator which converts Python functions to OpenAI-compatible JSON schema
- `ToolsRegistry` provides three built-in tools: `bash`, `read_file`, `write_file`
- Tool arguments are parsed from JSON strings to dictionaries

### Labeling Feature
- Agents can now have a `label` parameter for improved identification in logs
- The discovery agent uses `label="Discovery Agent"` and the main agent uses `label="Main Agent"`

### Uncommitted Changes
The current working copy has **unstaged modifications** to `examples/agentic_shell.py`:
- Updated discovery prompt to output to `.my_coding_agent/discovery.md`
- Added `read_file` and `write_file` tools to discovery agent
- Added discovery notes to main agent's system prompt
- Removed the line that appends discovery findings to main agent's messages (now uses file-based approach)

### Design Philosophy
- Minimal, focused codebase (~500 lines of core logic)
- Extensive logging with custom colored levels (TOOL, API, LLM)
- ASCII art startup banner with session info (model, branch, tools, session ID, timestamp)
- Focus on local LLM integration with OpenAI-compatible API

---

## 4. Available Tools

### `bash(command: str) -> str`
**Description:** Executes a shell command and returns stdout, stderr, and exit code.

**Usage:**
```python
result = bash("ls -la /Users/noordeepsingh/Workspace/my-coding-agent")
# Returns: "stdout:\n... \nexit_code: 0"
```

**Parameters:**
- `command` (str): Shell command to execute

**Returns:** Formatted string with stdout, stderr (if any), and exit code

**Use cases:** Running tests, installing packages, git operations, any shell task

---

### `read_file(path: str) -> str`
**Description:** Reads and returns the full contents of a file at the given path.

**Usage:**
```python
content = read_file("/Users/noordeepsingh/Workspace/my-coding-agent/pyproject.toml")
```

**Parameters:**
- `path` (str): Absolute path to the file

**Returns:** File contents as string, or error message if file not found

**Use cases:** Inspecting source code, configs, or any text file before editing

---

### `write_file(path: str, content: str) -> str`
**Description:** Writes content to a file, creating parent directories if needed.

**Usage:**
```python
result = write_file("/Users/noordeepsingh/Workspace/my-coding-agent/.my_coding_agent/discovery.md", content)
# Returns: "Written 1234 bytes to /Users/noordeepsingh/Workspace/my-coding-agent/.my_coding_agent/discovery.md"
```

**Parameters:**
- `path` (str): Absolute path for the file
- `content` (str): Content to write

**Returns:** Confirmation message with bytes written, or error message

**Use cases:** Creating new files or overwriting existing ones

---

## 5. Latest User Requests

The current user request is to **explore the workspace and create a discovery document** that:
1. Summarizes the current state of the codebase and git repository
2. Documents recent activities or changes
3. Provides insights and observations for context
4. Describes available tools and their usage
5. Captures the latest user requests
6. Includes recent conversation transcript

The goal is to create a **stable prompt prefix** for future conversations that provides continuity even when conversation history is truncated due to token limits.

---

## 6. Recent Conversation Transcript

### User Request (Current Session)
> "Explore the workspace and discover any relevant information that can help you understand the current state of the codebase, git repository, recent activities, and anything else that might be useful for future tasks. Summarize your findings in a concise way into a Markdown file saved to `.my_coding_agent/discovery.md` in the current workspace. It should contains: 1. a summary of the current state of the codebase and git repository, 2. any recent activities or changes that might be relevant, 3. any insights or observations that can help you understand the context better. 4. Tool descriptions and how to use them. 5. Latest user requests. 6. Recent transcript of the conversation. The goal is to create a Stable prompt prefix that can be prepended to future conversations to provide context and continuity, even if the conversation history gets truncated due to token limits."

### Tools Used in This Session
- `bash()` - Multiple commands to explore workspace structure, git history, and file contents
- `read_file()` - Reading source code files, configuration files, and documentation
- `write_file()` - Creating the discovery.md output file

### Key Actions Performed
1. Explored directory structure with `ls -la`
2. Retrieved git log history (last 20+ commits)
3. Checked git status (found unstaged changes)
4. Found and enumerated all Python source files
5. Read all core source modules (agent.py, llm.py, tools.py, logger.py, utils.py)
6. Read example code (agentic_shell.py)
7. Read project configuration (pyproject.toml, README.md)
8. Checked git remote configuration
9. Reviewed uncommitted changes in examples/agentic_shell.py

---

## Quick Reference

### Running the Agent
```bash
uv sync
uv run python examples/agentic_shell.py
```

### Starting a Local LLM Server
```bash
mlx-llm server --model qwen3:35b
# or
ollama serve
```

### Key Classes
- `LLM` - Base class for HTTP communication with LLM server
- `Agent` - Agent class with run loop and message management
- `ToolsRegistry` - Registry of available tools (bash, read_file, write_file)

### Key Methods
- `Agent.run(max_steps=5)` - Execute the agent loop
- `Agent.add_message(message)` - Add messages to conversation
- `LLM.chat_completion(messages, tools)` - Send to LLM and get response
- `LLM.execute_tool_calls(message)` - Execute tool calls from response

### Custom Log Levels
- `TOOL` (15) - Tool execution logs (magenta)
- `API` (25) - API communication logs (white)
- `LLM` (35) - LLM response logs (yellow)

---

*This document is designed to be a stable prompt prefix that can be prepended to future conversations to provide context and continuity.*

# Architecture

## Overview

`my-coding-agent` is a hand-rolled Python agent harness. There are no external agent frameworks — the entire agentic loop, tool dispatch, context management, and session persistence are implemented from scratch in ~1,000 lines across a small number of modules.

```
workflows/main.py           ← CLI entry point
│
├── agents/discovery.py     ← Discovery Agent (codebase mapping)
├── agents/session_analyzer.py  ← Session Analyzer Agent (post-run reporting)
│
└── src/my_coding_agent/
    ├── agent.py            ← Agent loop (extends LLM)
    ├── llm.py              ← LLM HTTP client + tool execution
    ├── tools.py            ← Tool registry and decorator
    ├── handoff.py          ← Context reset / handoff state transfer
    ├── logger.py           ← Custom logger, rich terminal output, TeeStream
    └── utils.py            ← Thin response parsing helpers
```

---

## Core Layers

### `LLM` (`llm.py`)

The base class. Owns the `httpx` session, calls `/v1/chat/completions`, and tracks every call in `self.llm_calls`. Key responsibilities:

- **`chat_completion(messages, tools, kind)`** — single POST to the LLM server; records token usage per call tagged by `kind` (`main`, `handoff`, `tool_router`, `tool_output_summarizer`, `tool_arg_correction`).
- **`route_tools(message, all_tools)`** — two-phase tool selection before each step: (1) keyword match on each tool's `tags`, (2) LLM fallback if phase 1 returns nothing outside the baseline. Baseline tools (`bash`, `read_file`, `read_tool_artifact`) are always included.
- **`execute_tool_calls(message)`** — iterates tool call requests in an LLM response, dispatches each through `invoke_tool`, collects results back as `role: tool` messages.
- **`invoke_tool(...)`** — calls the tool function with up to `_MAX_ARG_RETRIES` retries; on `TypeError` it asks the LLM to correct the arguments (`tool_arg_correction` call) before retrying. Recoverable exceptions are returned as error strings; non-recoverable ones re-raise.
- **Artifact separation** — when a tool returns a `(None, dict)` tuple (bash output or file content above `ARTIFACT_THRESHOLD`), the full dict is stored in `self.tool_artifacts[tool_call_id]` and an LLM-generated summary is sent back to the model instead. The full artifact is retrievable via the `read_tool_artifact` tool.
- **Arg aliases** — a static map remaps common model hallucinations (e.g. `bash(path=)` → `bash(command=)`) before dispatch.
- **Hooks** — `before_tool_call` and `after_tool_call` are user-injectable callbacks that wrap every tool dispatch.

### `Agent` (`agent.py`)

Extends `LLM`. Runs the main agentic loop in `run(max_steps)`.

Each step:
1. **Context pre-flight check** — computes `prompt_tokens / context_window`. If ≥ threshold (default 75%), triggers a context handoff (see below). If ≥ 100%, hard stops.
2. **Tool routing** — calls `route_tools` to select the relevant subset of tools for this step.
3. **LLM call** — `chat_completion(messages, routed_tools)`.
4. **Tool dispatch** — `execute_tool_calls(message)` appends tool result messages.
5. **Token tracking** — updates `last_prompt_tokens` for the next step's context check.
6. **Finish check** — stops on `finish_reason` of `stop`/`exit`/`quit` or when `step_num >= max_steps`.

After the loop: saves `session_data.json`, prints the run summary, detaches the session log.

### Context Handoff (`handoff.py` + `agent.py`)

When `prompt_tokens / context_window >= context_reset_threshold`, the agent:
1. Sends a structured handoff prompt asking the LLM to summarize progress, files changed, decisions made, and remaining work.
2. Saves the result as a markdown file under `.my_coding_agent/handoffs/`.
3. Saves session data and prints a summary for the current run.
4. Spawns a fresh `Agent` instance carrying only the system messages and the handoff as a user message — then calls `continuation.run(remaining_steps)` and returns its result.

This lets long-running tasks survive context exhaustion without silent truncation.

### `ToolsRegistry` (`tools.py`)

A plain class whose methods are the tools the LLM can call:

| Tool | Purpose |
|---|---|
| `bash(command)` | Runs a shell command; returns JSON `{stdout, stderr, exit_code, ok}` |
| `read_file(file_path)` | Reads a file; large files become artifacts |
| `write_file(file_path, content)` | Writes a file, creating parent dirs |
| `read_article(url)` | Fetches a URL and converts HTML → markdown |
| `read_tool_artifact(tool_call_id)` | Retrieves a previously stored large output |
| `delegate(task, context)` | Spawns a fresh read-only subagent for a focused task with the given context; returns its final report |

`delegate` recursively spawns a child `Agent` (read-only, `max_steps=5`, the `delegate` tool itself removed from its toolset), so the main agent can offload focused exploration without crowding its own context.

The `@tool` decorator (actually `function_to_json`) converts any `ToolsRegistry` method into an OpenAI-compatible tool definition by inspecting its signature and parsing Google-style docstrings for parameter descriptions and routing `Tags`.

### `Logger` (`logger.py`)

- Custom log levels: `TOOL` (15), `API` (25), `LLM` (35) between `DEBUG` and `INFO`.
- `DynamicStderrHandler` always writes to the live `sys.stderr`, so it follows the TeeStream replacement.
- `attach_session_log(path)` replaces `sys.stderr` with a `_TeeStream` that simultaneously writes to the original stderr, a plain log file, and an ANSI-colored log file under `.my_coding_agent/<session_id>/`.
- `print_banner` and `print_run_summary` render rich box-drawn terminal UIs including a `plotext` token consumption chart.

---

## Workflow Pipeline (`workflows/main.py`)

```
CLI (Click)
  │
  ├── 1. Discovery Agent (optional, --discover)
  │       Explores workspace → writes .my_coding_agent/discovery.md
  │
  ├── 2. Main Agent
  │       System prompt: workspace state + tool list + discovery.md pointer
  │       User prompt: task from --prompt / --interactive / default
  │       Tools: all ToolsRegistry methods
  │
  └── 3. Session Analyzer (optional, --analyze)
          Reads session_data.json + stderr.log → writes session_analysis.md
```

The system prompt for the Main Agent includes the current working directory, directory listing, git status, branch, and recent commits — all injected at startup.

---

## Session Persistence

Each run creates `.my_coding_agent/<session_id>/`:

| File | Contents |
|---|---|
| `stderr.log` | Plain-text log of the full run |
| `stderr_colored.log` | Same log with ANSI color codes |
| `session_data.json` | Metrics, tool records, LLM call log, stop reason |
| `tool_artifacts.json` | Full outputs for any call that triggered artifact separation |
| `session_analysis.md` | Post-run analysis report (if --analyze was used) |

Handoffs are saved under `.my_coding_agent/handoffs/`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | OpenAI-compatible API base URL |
| `OMLX_API_KEY` | `changeme` | API key (typically ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID |

The context window size is discovered at startup by querying `/v1/models` and reading `context_length` from the model's metadata. Falls back to 128k.

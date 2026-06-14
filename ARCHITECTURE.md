# Architecture

## Overview

`my-coding-agent` is a hand-rolled Python agent harness. There are no external agent frameworks — the entire agentic loop, tool dispatch, context management, and session persistence are implemented from scratch in ~1,000 lines across a small number of modules.

```
src/my_coding_agent/
├── workflows/main.py       ← CLI entry point
│
├── agents/discovery.py     ← Discovery Agent (codebase mapping)
├── agents/session_analyzer.py  ← Session Analyzer Agent (post-run reporting)
│
├── agent.py                ← Agent loop (holds an LLM client; delegates routing + execution)
├── llm.py                  ← LLM HTTP client (pure client)
├── tool_routing.py         ← ToolRouter (two-phase tool selection)
├── tool_execution.py       ← ToolExecutor (tool dispatch + artifacts)
├── tools.py                ← Tool registry and decorator
├── handoff.py              ← Context reset / handoff state transfer
├── logger/                 ← Logging, session-log capture, terminal UI (package)
│   ├── logging_core.py     ← Custom levels + ColoredFormatter + DynamicStderrHandler
│   ├── session_log.py      ← TeeStream + attach/detach_session_log
│   ├── banner.py           ← print_banner renderer + shared _git_branch
│   └── summary.py          ← print_run_summary renderer (+ token chart)
└── utils.py                ← Thin response parsing helpers
```

---

## Core Layers

### `LLM` (`llm.py`)

The pure HTTP client. Owns the `httpx` session, calls `/v1/chat/completions`, and tracks every call in `self.llm_calls`. Construction performs no network I/O — the model's context window is probed lazily on first access to `context_window`. Key responsibilities:

- **`chat_completion(messages, tools, kind)`** — single POST to the LLM server; records token usage per call tagged by `kind` (`main`, `handoff`, `tool_router`, `tool_output_summarizer`, `tool_arg_correction`).
- **`available_models` / `context_window`** — fetch the model list and resolve/cache the context window (128k fallback when unreachable).
- **`_request_with_retry`** — retries transient connection/timeout failures with backoff.
- **Hooks** — `before_tool_call` and `after_tool_call` callbacks are stored on the client (`_before_hook`/`_after_hook`); the `ToolExecutor` reads them when wrapping each dispatch.

Tool routing and tool execution are **not** on `LLM` — they live in the `ToolRouter` and `ToolExecutor` collaborators, each of which holds an `LLM` as its `client`.

### `ToolRouter` (`tool_routing.py`)

Holds the LLM client and selects the relevant tool subset for a message via **`route_tools(message, all_tools)`** — two-phase selection before each step: (1) keyword match on each tool's `tags`, (2) LLM fallback (`client.chat_completion(..., kind="tool_router")`) if phase 1 returns nothing outside the baseline. Baseline tools (`bash`, `read_file`, `read_tool_artifact`) are always included.

### `ToolExecutor` (`tool_execution.py`)

Holds the LLM client AND owns the `tool_artifacts` store (execution state). Runs every tool call requested in an LLM response:

- **`execute_tool_calls(message, conversation, tools)`** — iterates tool-call requests, dispatches each through `invoke_tool`, collects results back as `role: tool` messages. The `conversation` and `tools` are passed in explicitly (not read off the client) so the executor stays decoupled from the agent loop's state.
- **`invoke_tool(...)`** — calls the tool function with up to `_MAX_ARG_RETRIES` retries; on `TypeError` it asks the LLM to correct the arguments (`tool_arg_correction` call) before retrying. Recoverable exceptions are returned as error strings; non-recoverable ones re-raise.
- **Artifact separation** — when a tool returns a `(None, dict)` tuple (bash output or file content above `ARTIFACT_THRESHOLD`), the full dict is stored in `self.tool_artifacts[tool_call_id]` and an LLM-generated summary is sent back to the model instead. The full artifact is retrievable via the `read_tool_artifact` tool.
- **Arg aliases** — a static map remaps common model hallucinations (e.g. `bash(path=)` → `bash(command=)`) before dispatch; unknown kwargs are stripped to the tool's signature.

### `Agent` (`agent.py`)

Holds an `LLM` client via composition (`self.llm`) — it is **not** a subclass of `LLM`. In `__init__` it builds that client and the `ToolRouter(self.llm)` / `ToolExecutor(self.llm)` collaborators (all sharing the one client instance) to which it delegates routing and execution. Every client read goes through `self.llm.*` (`chat_completion`, `context_window`, `llm_calls`, `model`, `api_url`/`api_key`, the `_before_hook`/`_after_hook` hooks); agent-loop state (`messages`, `tools`, `step_num`, `tool_records`, `last_prompt_tokens`, …) stays on the agent. Runs the main agentic loop in `run(max_steps)`.

Each step:
1. **Context pre-flight check** — computes `prompt_tokens / context_window`. If ≥ threshold (default 75%), triggers a context handoff (see below). If ≥ 100%, hard stops.
2. **Tool routing** — calls `self._router.route_tools(signal, self.tools)` to select the relevant subset of tools for this step.
3. **LLM call** — `chat_completion(messages, routed_tools)`.
4. **Tool dispatch** — `self._executor.execute_tool_calls(message, self.messages, self.tools)` appends tool result messages.
5. **Token tracking** — updates `last_prompt_tokens` for the next step's context check.
6. **Finish check** — stops on `finish_reason` of `stop`/`exit`/`quit` or when `step_num >= max_steps`.

After the loop: saves `session_data.json` (including `self._executor.tool_artifacts`), prints the run summary, detaches the session log.

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

### `Logger` (`logger/` package)

Three independent concerns are split into focused submodules; the package
`__init__.py` re-exports the historical flat surface, so `from my_coding_agent.logger import get_logger, print_banner, ...` keeps working unchanged.

- **`logging_core.py`** — custom log levels `TOOL` (15), `API` (25), `LLM` (35) between `DEBUG` and `INFO`; the `ColoredFormatter`; the `DynamicStderrHandler` that always writes to the live `sys.stderr` (so it follows the TeeStream replacement); and the `get_logger` factory.
- **`session_log.py`** — `attach_session_log(path)` replaces `sys.stderr` with a `_TeeStream` that simultaneously writes to the original stderr, a plain log file, and an ANSI-colored log file under `.my_coding_agent/<session_id>/`; `detach_session_log` restores it.
- **`banner.py`** — `print_banner` renders the rich box-drawn startup banner (ASCII logo + run-metadata panel), plus the shared `_git_branch` helper used by both renderers.
- **`summary.py`** — `print_run_summary` renders the rich box-drawn end-of-run summary including a `plotext` token consumption chart, tool-call and context-reset sections; imports `_git_branch` from `banner`.

---

## Workflow Pipeline (`src/my_coding_agent/workflows/main.py`)

```
CLI (Click)
  │
  ├── 1. Discovery Agent (opt-in, default off; --discover)
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

The context window size is discovered lazily — on first access to `LLM.context_window` (at the start of `Agent.run`), not during construction — by querying `/v1/models` and reading `context_length` from the model's metadata. The result is cached; an unreachable server falls back to 128k. Constructing `LLM`/`Agent` therefore performs no network I/O.

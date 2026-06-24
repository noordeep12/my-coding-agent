# Architecture

## Overview

`my-coding-agent` is a hand-rolled Python agent harness. There are no external agent frameworks — the entire agentic loop, tool dispatch, context management, and session persistence are implemented from scratch.

```
src/my_coding_agent/
│
├── pipeline/                    ← Node-based DAG execution engine (package)
│   ├── __init__.py              ← Public surface: RunContext, Pipeline, AgentNode, build_default_pipeline
│   ├── context.py               ← RunContext dataclass: explicit data contract between nodes
│   ├── node.py                  ← Node protocol + BaseNode ABC
│   ├── dag.py                   ← Pipeline: ordered node list + step-loop execution engine
│   ├── nodes/                   ← One module per pipeline stage
│   │   ├── agent_node.py        ← AgentNode: session bookkeeping + pipeline runner (main entry)
│   │   ├── handoff.py           ← ContextHandoff: context reset state transfer
│   │   ├── router.py            ← ToolRouter: two-phase tool selection
│   │   ├── context_preflight.py ← ContextPreflightNode: context-window check + handoff trigger
│   │   ├── tool_routing.py      ← ToolRoutingNode: select relevant tool subset
│   │   ├── llm_call.py          ← LLMCallNode: chat_completion + append assistant message
│   │   ├── tool_dispatch.py     ← ToolDispatchNode: ToolExecutor.run() per step
│   │   ├── token_tracking.py    ← TokenTrackingNode: record step usage
│   │   └── finish_check.py      ← FinishCheckNode: detect stop/exit/quit finish reason
│   └── examples/
│       └── simple.py            ← CLI entry point (Click)
│
├── llm/                         ← LLM HTTP client (package)
│   └── __init__.py              ← LLM class, OMLX_* constants
├── tool_execution/              ← ToolExecutor + pure helpers (package)
│   ├── __init__.py              ← ToolExecutor: per-message run() (before/call/after)
│   ├── result_schema.py         ← Canonical envelope: build/validate/normalize
│   ├── args.py                  ← Tool-call parse + alias remap + kwarg strip
│   └── output.py                ← Truncation + artifact description
├── tool_registry/               ← ToolRegistry class + tool definition converter (package)
│   ├── __init__.py              ← Re-export facade (ToolRegistry, tool)
│   ├── converter.py             ← function_to_json + tool decorator
│   └── registry.py             ← ToolRegistry: callable tool methods
├── logger/                      ← Logging, session-log capture, terminal UI (package)
│   ├── __init__.py              ← Re-export facade
│   ├── logging_core.py          ← Custom levels + ColoredFormatter
│   ├── session_log.py           ← TeeStream + attach/detach_session_log
│   ├── banner.py                ← print_banner renderer
│   └── summary.py               ← print_run_summary renderer
├── observability/               ← Structured session capture (package)
│   ├── recorder.py              ← Recorder: events.jsonl writer + contextvars
│   ├── records.py               ← Call-record builders
│   └── events.py                ← Typed schema (Session/LLMCall/ToolCall/Handoff)
└── utils/                       ← Shared helpers (package)
    ├── __init__.py
    ├── exceptions.py            ← MyCodingAgentError hierarchy
    └── parsing.py               ← Response-parsing helpers (extract_message, etc.)
```

---

## Core Layers

### `LLM` (`llm/`)

The pure HTTP client. Owns the `httpx` session, calls `/v1/chat/completions`, and tracks every call in `self.llm_calls`. Construction performs no network I/O — the model's context window is probed lazily on first access to `context_window`. Key responsibilities:

- **`chat_completion(messages, tools, kind)`** — single POST to the LLM server; records token usage per call tagged by `kind` (`main`, `handoff`, `tool_router`, `tool_output_summarizer`, `tool_arg_correction`).
- **`available_models` / `context_window`** — fetch the model list and resolve/cache the context window (128k fallback when unreachable).
- **`_request_with_retry`** — retries transient connection/timeout failures with backoff.

### `ToolRouter` (`pipeline/nodes/router.py`)

Holds the LLM client and selects the relevant tool subset for a message via **`route_tools(message, all_tools)`** — two-phase selection before each step: (1) keyword match on each tool's `tags`, (2) LLM fallback if phase 1 returns nothing outside the baseline. Baseline tools (`bash`, `read_file`, `read_tool_artifact`) are always included.

### `ToolExecutor` (`tool_execution/` package)

Constructed **per assistant message** (`ToolExecutor(message, llm)`). Runs `before_tool_call` → `invoke_tool` → `after_tool_call` per call. Returns tool messages and records. Normalizes all results into the canonical `{schema_version, tool, ok, output, error, metadata}` envelope.

### `Pipeline` (`pipeline/` package)

The node-based DAG execution engine.

**`RunContext` (`context.py`)** — the explicit data contract that flows through the pipeline. Holds immutable run config (session id, max steps, LLM client, recorder, all tools) and mutable state fields (messages, step_num, last_prompt_tokens, tool_records, tool_artifacts, handoff_records). Control signals (`signal`, `stop_reason`) are written by nodes and read by `Pipeline.execute`.

**`Node` protocol + `BaseNode` (`node.py`)** — a `Node` is any callable with a `name: str` and a `run(ctx: RunContext) -> None` method. Nodes read and write `ctx` in place.

**`Pipeline` (`dag.py`)** — takes an ordered list of `Node` objects. `run_step` executes every node in order for one step, short-circuiting when any node sets a non-`CONTINUE` signal. `execute` wraps `run_step` in the outer step loop.

**The six default nodes** (instantiated by `build_default_pipeline()`):

| Node | Stage | What it does |
|---|---|---|
| `ContextPreflightNode` | 1 | Checks `last_prompt_tokens / context_window`; sets STOP (limit), RESET (handoff), or CONTINUE |
| `ToolRoutingNode` | 2 | Creates `ToolRouter(ctx.llm)`, calls `route_tools`, writes `ctx.routed_tools` |
| `LLMCallNode` | 3 | Increments `step_num`, calls `chat_completion`, appends assistant message |
| `ToolDispatchNode` | 4 | Builds `ToolExecutor(last_message, ctx.llm)`, runs it, merges records and artifacts into `ctx` |
| `TokenTrackingNode` | 5 | Reads `ctx.last_response`, updates `ctx.last_prompt_tokens`, logs usage |
| `FinishCheckNode` | 6 | Reads finish_reason from `ctx.last_response`; sets STOP on stop/exit/quit |

### `AgentNode` (`pipeline/nodes/agent_node.py`)

The top-level entry point. Holds an `LLM` client via composition (`self.llm`) — not a subclass. `__init__` builds the client, assigns a session id, attaches the session log, and initializes run stats.

- **`execute(max_steps)`** — stand-alone entry: constructs a `RunContext`, builds the pipeline via `build_default_pipeline(spawn_fn=...)`, delegates to `pipeline.execute(ctx)`, saves session data, and prints the summary.
- **`run(ctx)`** — embedded entry: runs `execute()` and writes results back to the provided `RunContext`; used when `AgentNode` is a step in a larger outer pipeline.

`AgentNode` owns session bookkeeping: banner printing, session log attachment/detachment, `session_data.json` + `tool_artifacts.json` persistence, run summary, and observability recorder start/finish. It also owns `_generate_handoff`, `_spawn_continuation`, and `_handle_context_reset` — the context-reset machinery called by `ContextPreflightNode` via the `spawn_fn` closure.

### Context Handoff (`pipeline/nodes/handoff.py` + `pipeline/nodes/context_preflight.py`)

When `ContextPreflightNode` detects that `prompt_tokens / context_window >= context_reset_threshold`, it calls `spawn_fn()` — a closure set by `AgentNode.execute` that calls `AgentNode._handle_context_reset`. That method:
1. Sends a structured handoff prompt asking the LLM to summarize progress, files changed, decisions made, and remaining work.
2. Saves the result as a markdown file under `.my_coding_agent/handoffs/`.
3. Saves session data and prints a summary for the current run.
4. Spawns a fresh `AgentNode` instance carrying only the system messages and the handoff as a user message.

`ContextPreflightNode` stores the result in `ctx.continuation_messages` and sets `ctx.signal = "RESET"`, which causes `Pipeline.execute` to return those messages immediately.

### `ToolRegistry` (`tool_registry/` package)

A plain class whose methods are the tools the LLM can call:

| Tool | Purpose |
|---|---|
| `bash(command)` | Runs a shell command; reports `stdout, stderr, exit_code, ok` |
| `read_file(file_path)` | Reads a file; large files become artifacts |
| `write_file(file_path, content)` | Writes a file, creating parent dirs |
| `read_article(url)` | Fetches a URL and converts HTML → markdown |
| `read_tool_artifact(tool_call_id)` | Retrieves a previously stored large output |
| `delegate(task, context)` | Spawns a fresh read-only subagent for a focused task |

The `@tool` decorator converts any `ToolRegistry` method into an OpenAI-compatible tool definition by inspecting its signature and parsing Google-style docstrings.

### `Logger` (`logger/` package)

- **`logging_core.py`** — custom log levels `TOOL` (15), `API` (25), `LLM` (35); `ColoredFormatter`; `get_logger`.
- **`session_log.py`** — `attach_session_log(path)` / `detach_session_log` tee stderr to per-session log files.
- **`banner.py`** — `print_banner` renders the startup banner.
- **`summary.py`** — `print_run_summary` renders the end-of-run summary with token chart.

### `Observability` (`observability/` package)

A capture layer that writes a per-session `events.jsonl`.

- **`recorder.py`** — `Recorder` appends events (LLM calls, tool I/O, handoffs, agent links) as newline-delimited JSON. Two `ContextVar`s (`current_session_id`, `current_recorder`) let delegated subagents record their parent link.
- **`records.py`** — `error_record()` / `call_record()` build the per-tool-call dicts appended to `tool_records` and persisted in `session_data.json`.
- **`events.py`** — typed schema: `Session`, `LLMCall`, `ToolCall`, `Handoff`.

---

## Entry Point (`pipeline/examples/simple.py`)

```
CLI (Click)
  │
  └── AgentNode
        System prompt: workspace state + tool list
        User prompt: task from --prompt / --interactive / default
        Tools: all ToolRegistry methods
```

---

## Session Persistence

Each run creates `.my_coding_agent/<session_id>/`:

| File | Contents |
|---|---|
| `stderr.log` | Plain-text log of the full run |
| `stderr_colored.log` | Same log with ANSI color codes |
| `session_data.json` | Metrics, tool records, LLM call log, stop reason |
| `events.jsonl` | Structured observability event stream |
| `tool_artifacts.json` | Full outputs for large tool results |

Handoffs are saved under `.my_coding_agent/handoffs/`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | OpenAI-compatible API base URL |
| `OMLX_API_KEY` | `changeme` | API key |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-4bit` | Model ID |

The context window size is discovered lazily on first access to `LLM.context_window`.

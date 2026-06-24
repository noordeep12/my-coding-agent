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
├── tool_execution/         ← ToolExecutor + its pure helpers (package)
│   ├── __init__.py         ← ToolExecutor: per-message run() (before/call/after)
│   ├── result_schema.py    ← Canonical envelope: build/validate/normalize (pure)
│   ├── args.py             ← Tool-call parse + alias remap + kwarg strip (pure)
│   └── output.py           ← Truncation + deterministic artifact description
├── tool_registry/          ← ToolRegistry class + OpenAI tool-definition converter (package)
│   ├── __init__.py         ← Re-export facade (ToolRegistry, ARTIFACT_THRESHOLD, function_to_json, tool)
│   ├── converter.py        ← function_to_json + tool + private docstring parsers (pure)
│   └── registry.py        ← ToolRegistry: callable tool methods + ARTIFACT_THRESHOLD
├── handoff.py              ← Context reset / handoff state transfer
├── logger/                 ← Logging, session-log capture, terminal UI (package)
│   ├── __init__.py         ← Re-export facade (get_logger, print_banner, attach/detach_session_log, …)
│   ├── logging_core.py     ← Custom levels + ColoredFormatter + DynamicStderrHandler
│   ├── session_log.py      ← TeeStream + attach/detach_session_log
│   ├── banner.py           ← print_banner renderer + shared _git_branch
│   └── summary.py          ← print_run_summary renderer (+ token chart)
├── observability/          ← Structured session capture + HTML tree viewer (package)
│   ├── recorder.py         ← Recorder: append-as-you-go events.jsonl writer + contextvars
│   ├── records.py          ← Call-record builders (per-tool-call dicts for tool_records)
│   ├── events.py           ← Typed schema (Session/LLMCall/ToolCall/Handoff/TreeNode)
│   ├── reader.py           ← load_session(s) + derived analytical views
│   ├── tree.py             ← build_trace_tree: reconstruct the hierarchical pipeline tree
│   ├── pricing.py          ← Per-model token→USD table (defaults free for local models)
│   └── report.py           ← Generates a self-contained HTML/JS viewer (no deps, offline)
└── utils.py                ← Thin response parsing helpers
```

---

## Core Layers

### `LLM` (`llm.py`)

The pure HTTP client. Owns the `httpx` session, calls `/v1/chat/completions`, and tracks every call in `self.llm_calls`. Construction performs no network I/O — the model's context window is probed lazily on first access to `context_window`. Key responsibilities:

- **`chat_completion(messages, tools, kind)`** — single POST to the LLM server; records token usage per call tagged by `kind` (`main`, `handoff`, `tool_router`, `tool_output_summarizer`, `tool_arg_correction`).
- **`available_models` / `context_window`** — fetch the model list and resolve/cache the context window (128k fallback when unreachable).
- **`_request_with_retry`** — retries transient connection/timeout failures with backoff.
- **Recorder** — the optional observability `Recorder` is attached as `llm._recorder` by the `Agent`; the `ToolExecutor` calls it directly to capture tool I/O. (The old `_before_hook`/`_after_hook` callbacks were removed.)

Tool routing and tool execution are **not** on `LLM` — they live in the `ToolRouter` and `ToolExecutor` collaborators, each of which holds an `LLM` as its `client`.

### `ToolRouter` (`tool_routing.py`)

Holds the LLM client and selects the relevant tool subset for a message via **`route_tools(message, all_tools)`** — two-phase selection before each step: (1) keyword match on each tool's `tags`, (2) LLM fallback (`client.chat_completion(..., kind="tool_router")`) if phase 1 returns nothing outside the baseline. Baseline tools (`bash`, `read_file`, `read_tool_artifact`) are always included.

### `ToolExecutor` (`tool_execution/` package)

A package split by responsibility. The `ToolExecutor` class (`__init__.py`) is
constructed **per assistant message** (`ToolExecutor(message, llm)`) and holds
that message's `tool_calls` plus the running `tool_messages` / `tool_records` it
fills, the `tool_artifacts` it offloads, and its own `ToolRegistry`. **It makes
no LLM calls** — the `llm` is kept only for the session log path and the
observability recorder (`llm._recorder`). It **composes** three pure sibling
modules (no client, no state, no I/O) by calling their functions directly:

- **`result_schema.py`** — the canonical-envelope contract: `build_tool_result()`,
  `validate_tool_result()`, and `result_envelope()` (source-shape normalizer).
- **`args.py`** — `parse_tool_call()`, `apply_arg_aliases()`, `strip_unknown_args()`.
- **`output.py`** — `validate_tool_output()` (truncation) and `describe_artifact()`
  (the deterministic, no-LLM artifact description). `_extract_summary()` and
  `summarize_artifact()` (an LLM summary) live here too but are **currently
  unused** by the executor, kept for a future opt-in.

The methods each do real work (no pass-through delegators), split around the
function call into three phases — **before → call → after** — driven by `run()`:

- **`run()`** — iterates `self.tool_calls`, runs `before_tool_call` → `invoke_tool` → `after_tool_call` per call (parse failures short-circuit to an error result), and appends each `role: tool` message + record. Returns the two lists (also held as attributes).
- **`before_tool_call(func_name, args)`** — *before the call*: alias-remaps args (`args.apply_arg_aliases`), strips unknown kwargs, and stamps the recorder's start time. Returns the prepared args.
- **`invoke_tool(tool_call_id, func_name, args)`** — *the call only*: invokes `func_name(**args)` against `self.registry`. Returns `(raw_result, failure)` — the tool's raw return and `None` on success, or `None` and a `{reason, error}` descriptor on a handled failure (not-found, wrong-args, recoverable raise). **No retries, no LLM**: a wrong-argument call fails directly; non-recoverable exceptions re-raise. It builds no envelope or record.
- **`after_tool_call(tool_call_id, func_name, args, raw_result, failure)`** — *after the call*: turns the raw result (or failure) into the final `(content, status, record)` — offload + deterministically describe artifact tuples → coerce → truncate → normalize to the canonical envelope (success) or build the error envelope from the descriptor (failure), build the record via `observability.records`, serialize, and let the recorder capture the final content (a nested `capture` helper calling `llm._recorder.after_tool`).
- **Canonical tool-output schema** — the single choke point that enforces a uniform contract on *every* result that reaches the agent (direct results, described artifacts, parse/arg errors, raised exceptions). Each is normalized via `result_schema` into one envelope — `{schema_version, tool, ok, output, error, metadata}`, modeled on bash's `ok`/`exit_code` — and checked with `validate_tool_result()` before being serialized to JSON for the model. `output` carries the raw payload (stdout / file content / report / description); tool-specific extras (`exit_code`, `stderr`, `file_path`, `artifact`, `truncated`, …) live in the flexible `metadata` bag. Failure is detected from bash-style `ok`/`exit_code`, the stored artifact, the `Error…` string convention, or an exception, giving the agent **and** the viewer one consistent success/failure signal.
- **Artifact separation** — when a tool returns a `(None, dict)` tuple (bash output above `ARTIFACT_THRESHOLD`), the full dict is stored in `self.tool_artifacts[tool_call_id]` and a deterministic description (`describe_artifact`: exit status + byte counts, or file metadata, plus a `read_tool_artifact` pointer) is sent back instead — no LLM. The full artifact is retrievable via the `read_tool_artifact` tool. Note: a per-message executor resets `tool_artifacts`, so cross-step retrieval does not persist (the agent aggregates them for `session_data.json`).
- **Arg aliases** — a static map remaps common model hallucinations (e.g. `bash(path=)` → `bash(command=)`) before dispatch; unknown kwargs are stripped to the tool's signature.

### `Agent` (`agent.py`)

Holds an `LLM` client via composition (`self.llm`) — it is **not** a subclass of `LLM`. In `__init__` it builds that client and the `ToolRouter(self.llm)` collaborator; the `ToolExecutor` is built **per step** inside the loop (`ToolExecutor(message, self.llm)`). Every client read goes through `self.llm.*` (`chat_completion`, `context_window`, `llm_calls`, `model`, `api_url`/`api_key`); agent-loop state (`messages`, `tools`, `step_num`, `tool_records`, `tool_artifacts`, `last_prompt_tokens`, …) stays on the agent — `tool_artifacts` accumulates each step's executor artifacts for `session_data.json`. Runs the main agentic loop in `run(max_steps)`.

Each step:
1. **Context pre-flight check** — computes `prompt_tokens / context_window`. If ≥ threshold (default 75%), triggers a context handoff (see below). If ≥ 100%, hard stops.
2. **Tool routing** — calls `self._router.route_tools(signal, self.tools)` to select the relevant subset of tools for this step.
3. **LLM call** — `chat_completion(messages, routed_tools)`.
4. **Tool dispatch** — builds `ToolExecutor(message, self.llm)` and calls `executor.run()`, appends its tool-result messages, and merges `executor.tool_artifacts` into `self.tool_artifacts`.
5. **Token tracking** — updates `last_prompt_tokens` for the next step's context check.
6. **Finish check** — stops on `finish_reason` of `stop`/`exit`/`quit` or when `step_num >= max_steps`.

After the loop: saves `session_data.json` (including the accumulated `self.tool_artifacts`), prints the run summary, detaches the session log.

### Context Handoff (`handoff.py` + `agent.py`)

When `prompt_tokens / context_window >= context_reset_threshold`, the agent:
1. Sends a structured handoff prompt asking the LLM to summarize progress, files changed, decisions made, and remaining work.
2. Saves the result as a markdown file under `.my_coding_agent/handoffs/`.
3. Saves session data and prints a summary for the current run.
4. Spawns a fresh `Agent` instance carrying only the system messages and the handoff as a user message — then calls `continuation.run(remaining_steps)` and returns its result.

This lets long-running tasks survive context exhaustion without silent truncation.

### `ToolRegistry` (`tool_registry/` package)

A plain class whose methods are the tools the LLM can call:

| Tool | Purpose |
|---|---|
| `bash(command)` | Runs a shell command; reports `stdout, stderr, exit_code, ok` |
| `read_file(file_path)` | Reads a file; large files become artifacts |
| `write_file(file_path, content)` | Writes a file, creating parent dirs |
| `read_article(url)` | Fetches a URL and converts HTML → markdown |
| `read_tool_artifact(tool_call_id)` | Retrieves a previously stored large output |
| `delegate(task, context)` | Spawns a fresh read-only subagent for a focused task with the given context; returns its final report |

Each tool returns its natural value (a string, or bash's `{stdout, stderr, exit_code, ok}`); the **`ToolExecutor` normalizes all of them into the canonical tool-output envelope** (see above) before the agent sees them — bash's fields fold into `output`/`metadata`, plain strings become `output`. Tools therefore stay simple and need not know about the schema.

`delegate` recursively spawns a child `Agent` (read-only, `max_steps=5`, the `delegate` tool itself removed from its toolset), so the main agent can offload focused exploration without crowding its own context.

The `@tool` decorator (actually `function_to_json`) converts any `ToolRegistry` method into an OpenAI-compatible tool definition by inspecting its signature and parsing Google-style docstrings for parameter descriptions and routing `Tags`.

### `Logger` (`logger/` package)

Three independent concerns are split into focused submodules; the package
`__init__.py` re-exports the historical flat surface, so `from my_coding_agent.logger import get_logger, print_banner, ...` keeps working unchanged.

- **`logging_core.py`** — custom log levels `TOOL` (15), `API` (25), `LLM` (35) between `DEBUG` and `INFO`; the `ColoredFormatter`; the `DynamicStderrHandler` that always writes to the live `sys.stderr` (so it follows the TeeStream replacement); and the `get_logger` factory.
- **`session_log.py`** — `attach_session_log(path)` replaces `sys.stderr` with a `_TeeStream` that simultaneously writes to the original stderr, a plain log file, and an ANSI-colored log file under `.my_coding_agent/<session_id>/`; `detach_session_log` restores it.
- **`banner.py`** — `print_banner` renders the rich box-drawn startup banner (ASCII logo + run-metadata panel), plus the shared `_git_branch` helper used by both renderers.
- **`summary.py`** — `print_run_summary` renders the rich box-drawn end-of-run summary including a `plotext` token consumption chart, tool-call and context-reset sections; imports `_git_branch` from `banner`.

### `Observability` (`observability/` package)

A separate **capture layer** plus a **viewer**, added so a run can be audited
post-session without reading flat logs. It is independent of `logger/` (which is
left untouched) and additive to the existing `session_data.json`.

- **`recorder.py`** — the `Recorder` writes a per-session `events.jsonl`
  (newline-delimited, flushed per row, so a crashed run still leaves a
  diagnosable trail). Wired in at four points: a single **LLM choke point** in
  `llm.chat_completion` (the `Agent` sets `self.llm._recorder`) records every call
  kind with latency, tokens, the input conversation snapshot — kept for **all**
  chat-completion kinds (including the ancillary `tool_router` /
  `tool_output_summarizer` / `tool_arg_correction`) via `FULL_PAYLOAD_KINDS` so
  each call's input/output is inspectable like the main call — and the
  response; the `ToolExecutor` calls **`Recorder.before_tool`/`after_tool`
  directly** (via `llm._recorder`, not the client hooks) to capture each tool's
  I/O and latency; `ToolRouter._finish_route` calls
  `record_router` with the **selected tool subset** per step; and `record_handoff`
  captures context-reset events. Two `ContextVar`s set in `Agent.run`
  (`current_session_id`, `current_recorder`) let a delegated subagent record its
  `parent_session_id` and let `delegate` attach the spawned child's id to its tool
  call (`note_delegate_child`) for an exact parent→child link.
- **`records.py`** — `error_record()` / `call_record()` build the per-tool-call
  dicts (`name`/`args`/`ok`/`status`/…) the `ToolExecutor` appends to
  `Agent.tool_records` and persists in `session_data.json`. Owning the record
  shape here keeps it consistent across success, failure, skip, and parse-error
  paths.
- **`events.py`** — the reader-side typed schema (`Session`, `LLMCall`,
  `ToolCall`, `Handoff`) plus `TreeNode`, the node type for the trace tree.
- **`reader.py`** — `load_session` / `load_sessions_by_id` / `load_all_sessions`
  parse `events.jsonl` (joining `ok`/`status` back from `session_data.json`) and
  derive views: context-growth series, message diffs, cost/latency bottlenecks,
  loop/duplicate detection, and extracted code blocks.
- **`tree.py`** — `build_trace_tree` reconstructs the hierarchical-by-step
  pipeline tree from the event timeline: `Agent` → steps → `Agent._context_preflight`
  / `ToolRouter.route_tools` / `LLM.chat_completion` (its reasoning/content fold
  into `output`) / one `ToolExecutor.invoke_tool: <name>` per tool call, nesting
  delegated subagents under their `delegate` ToolExecutor. **Titles are the real
  `Class.method` executed** (so the tree reads as the call stack), and the
  **ancillary LLM calls are nested as their own `LLM.chat_completion` nodes** —
  the routing fallback under `ToolRouter.route_tools`, and
  `ToolExecutor._summarize_artifact` / `._correct_args` under their tool — each
  with full input/output, but excluded from the agent's context-window accounting
  since they run on their own ephemeral conversation. Each node carries
  `status`/`message`/`input`/`output` metadata; a tool is flagged **failure** (red
  error logo) when it raises *or* its result envelope reports `ok:false`
  (`_result_failed`), so bash-style failures show like any other. `annotate_context`
  adds a per-node context-window bar (`ctx`: history + tokens added/removed +
  owning agent name/id, re-anchored to each call's exact `prompt_tokens`).
- **`pricing.py`** — an editable `{model: (in_$, out_$)}` table; local models
  resolve to `$0.00`.
- **`report.py`** — `write_report` renders the sessions (via `reader`/`tree`) into
  a single self-contained `.my_coding_agent/viewer.html`: inline CSS + vanilla JS
  with the trace data embedded as JSON (the `</script>` close is escaped), so it
  works offline with no server and no dependencies. The page is a two-pane viewer
  — a collapsible, searchable trace tree (left) and the selected object's metadata
  (right). A synthetic **Session Overview** node (top of the tree) surfaces the
  derived views from `reader` — context-growth chart, per-step bottlenecks table,
  and loop/redundancy list — while the sticky header shows at-a-glance cost / token
  / failure chips; each LLM/tool row shows a tokens-added badge, the right pane has
  a **CONTEXT WINDOW** section, and every input/output box has a copy button.
  Exposed as the `my-coding-agent-viewer` console script.

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
  │       Tools: all ToolRegistry methods
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
| `events.jsonl` | Structured observability event stream (LLM calls, tool I/O, handoffs, agent links) — read by the viewer |
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

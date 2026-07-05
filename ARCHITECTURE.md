# Architecture

## Overview

`my-coding-agent` is a hand-rolled Python agent harness. There are no external agent frameworks — the entire agentic loop, tool dispatch, context management, and session persistence are implemented from scratch.

```
src/my_coding_agent/
│
├── engine/                      ← Owns execution: LLM client, tools, and AgentNode
│   ├── __init__.py              ← Public surface: AgentNode, LLM, ToolRegistry, tool
│   ├── agent.py                 ← AgentNode: session bookkeeping + pipeline runner (main entry)
│   ├── schema.py                ← Engine event type constants (SESSION_START, LLM_CALL, etc.)
│   ├── routing.py               ← ToolRouter: two-phase tool selection
│   ├── llm/                     ← LLM HTTP client
│   │   ├── __init__.py          ← LLM class, OMLX_* constants
│   │   └── schema.py            ← LLM request/response shape constants
│   ├── tool_execution/          ← ToolExecutor + pure helpers
│   │   ├── __init__.py          ← ToolExecutor: per-message run() (before/call/after; overlaps read-only groups)
│   │   ├── concurrency.py       ← Read-only independence gate + bounded worker ceiling (issue #65)
│   │   ├── schema.py            ← Canonical envelope shapes (version, keys, preview descriptor)
│   │   ├── envelope.py          ← Envelope builders: build/validate/normalize
│   │   ├── args.py              ← Tool-call parse + alias remap + kwarg strip
│   │   ├── output.py            ← Truncation + artifact preview builder (bounded excerpt + skim guidance)
│   │   ├── lang.py              ← Per-box language hint (metadata.lang: command/output/error)
│   │   └── records.py           ← Call-record builders (error_record, call_record)
│   └── tool_registry/           ← ToolRegistry class + tool definition converter
│       ├── __init__.py          ← Re-export facade (ToolRegistry, tool)
│       ├── converter.py         ← function_to_json + tool decorator
│       ├── registry.py          ← ToolRegistry methods + artifact_file_path (path scheme SoT)
│       └── schema.py            ← Tool definition JSON shape constants
│
├── pipeline/                    ← Pure DAG building and execution
│   ├── __init__.py              ← Public surface: RunContext, Pipeline, build_default_pipeline
│   ├── context.py               ← RunContext dataclass: explicit data contract between nodes
│   ├── node.py                  ← Node protocol + BaseNode ABC
│   ├── dag.py                   ← Pipeline: ordered node list + step-loop execution engine
│   ├── schema.py                ← Pipeline typed contracts (ROUTER constant + ContextHandoff)
│   ├── nodes/                   ← One module per pipeline stage (one node per file)
│   │   ├── context_guard.py     ← ContextGuardNode: context-window budget check + handoff trigger
│   │   ├── context_summarizer.py ← ContextSummarizerNode: triggered full-conversation summarization (report/handoff)
│   │   ├── tool_routing.py      ← ToolRoutingNode: select relevant tool subset
│   │   ├── llm_call.py          ← LLMCallNode: chat_completion + append assistant message
│   │   ├── tool_dispatch.py     ← ToolDispatchNode: ToolExecutor.run() per step
│   │   ├── anomaly_detect.py    ← AnomalyDetectNode: detects same-class tool-failure streaks, detection-only
│   │   └── finalize_step.py     ← FinalizeStepNode: record step usage, detect stop/exit/quit + step-ceiling cutoff
│   ├── anomaly.py               ← Pure helpers: error_signature(), trailing_streak(), STREAK_THRESHOLD
│   └── examples/
│       └── simple.py            ← CLI entry point (Click)
│
├── observability/               ← Passive event capture (never controls execution)
│   ├── __init__.py              ← Re-export facade (Recorder, current_session_id, current_recorder)
│   ├── recorder.py              ← Recorder: events.jsonl writer + event type constants + contextvars
│   └── schema.py                ← JSONL event row shape constants
│
├── viewer/                      ← Active read-side: parse events.jsonl + serve browser UI
│   ├── __init__.py              ← Public surface: TraceNode, TraceSession, load_session, list_sessions
│   ├── schema.py                ← TraceNode + TraceSession dataclasses (typed contracts)
│   ├── pricing.py               ← Model price table + compute_cost() helper
│   ├── reader.py                ← Parse events.jsonl → flat TraceSession; ctx-window deltas, loop detection, analytics
│   ├── server.py                ← Localhost HTTP server + embedded single-page Trace Explorer UI (Preact + htm)
│   └── _vendor/                 ← Offline-vendored UI libs (Preact, hooks, htm) — no CDN
│
└── utils/                       ← Generic helpers
    ├── __init__.py              ← Re-export facade (get_logger, print_banner, etc.)
    ├── exceptions.py            ← MyCodingAgentError hierarchy
    ├── parsing.py               ← Response-parsing helpers (extract_message, etc.)
    ├── logging_core.py          ← Custom levels, ColoredFormatter, TeeStream, attach/detach_session_log
    └── terminal_ui.py           ← print_banner + print_run_summary renderers, _git_branch
```

---

## Core Layers

### `engine/` — Execution Owner

The engine package owns all execution concerns: the LLM HTTP client, tool dispatch, tool definitions, and the top-level `AgentNode` that drives the agentic loop.

### `LLM` (`engine/llm/`)

The pure HTTP client. Owns the `httpx` session, calls `/v1/chat/completions`, and tracks every call in `self.llm_calls`. Construction performs no network I/O — the model's context window is probed lazily on first access to `context_window`. Key responsibilities:

- **`chat_completion(messages, tools, kind)`** — single POST to the LLM server; records token usage per call tagged by `kind` (`main`, `handoff`, `report`, `tool_router`, `tool_arg_correction`, `artifact_query`).
- **`available_models` / `context_window`** — fetch the model list and resolve/cache the context window (128k fallback when unreachable).
- **`_request_with_retry`** — retries transient connection/timeout failures with backoff.

### `ToolRouter` (`engine/routing.py`)

Holds the LLM client and answers, per call, which tools match a signal via **`route_tools(message, all_tools, has_previous_selection=False)`**, returning `(selected_or_None, phase)`. Phase 1 is a zero-cost, whole-word (`\b`-bounded, case-insensitive) match on each tool's `tags` — a substring like `"file"` no longer matches inside `"profile"`. Phase 2 (one LLM call) fires only on a cold start (`has_previous_selection=False`) with no tag match anywhere; a mid-run no-match instead returns `(None, "carry_forward")` so the caller reuses its last selection without an extra LLM call. Baseline tools (`bash`, `read_file`, `read_tool_artifact`) are always included. `ToolRouter` is stateless per call and does no recording — event emission is the caller's decision (see `ToolRoutingNode` below). Consumed only via `engine.routing` (e.g. `pipeline/nodes/tool_routing.py`); `pipeline/__init__.py` does not re-export it, keeping the `engine` → `pipeline` dependency one-way per the layered-dependency rule.

### `ToolExecutor` (`engine/tool_execution/` package)

Constructed **per assistant message** (`ToolExecutor(message, llm, tools=ctx.all_tools)`). Runs `before_tool_call` → `invoke_tool` → `after_tool_call` per call. Returns tool messages and records. Normalizes all results into the canonical `{schema_version, tool, ok, output, error, metadata}` envelope. For a `bash` result each datum maps to exactly one field — `stdout`→`output`, `stderr`→`error` (whenever non-empty, **regardless of `ok`**, so a masked pipeline failure surfaces as `ok:true` with a non-null `error`), `exit_code`→`metadata.exit_code`, success→`ok` — with nothing duplicated (`ok` is the sole success verdict; a non-null `error` does not imply failure). `result_envelope` only handles the single live artifact shape (a dict artifact carrying a preview) — no legacy fallbacks for shapes no tool produces. The bash-style `ok`/`stdout`/`stderr`/`exit_code` sniff is gated by tool name (`bash` only), not by content shape, so a verbatim fetched JSON body that happens to contain an `ok` key is never reinterpreted as that contract. A structured-return tool's own `metadata` bag (e.g. `fetch_web`'s `{content_type, transform}`) is merged into the envelope's `metadata` untouched; `metadata.artifact`/`tool_call_id` are set only when a stream was actually offloaded (non-empty `preview`), so a small structured return is never mislabeled as offloaded. Every envelope also carries an advisory `metadata.lang = {command, output, error}` language hint (`lang.resolve_lang`, values `python`/`shell`/`json`/`text`) resolved from the tool name and args — `bash` command→`shell`, a `read_file` output→its file extension, else a JSON sniff, else `text` — so the trace viewer highlights each box without re-inferring; it never changes `output`/`error`/`ok`/`exit_code`. It makes no LLM calls itself — the LLM client is held only for the session log path and the observability recorder. Forwards the run's toolset **and its `llm` client** to the `ToolRegistry` so toolset-aware tools (notably `delegate`) and `read_tool_artifact`'s extraction call can use them.

**Concurrent dispatch (issue #65).** The tool calls in one assistant message are parsed, then partitioned by `_plan_groups` into ordered execution groups: a maximal run of *contiguous* provably-read-only calls becomes one group the executor **overlaps** (bounded `ThreadPoolExecutor`), while every other call — a parse error, or one whose effects can't be proven read-only — is its own singleton group run inline in sequence. A non-overlappable call is therefore a **barrier**: nothing after it starts until it finishes, and it starts only once everything before it finished, so the observable ordering of the sequential path is preserved for every non-overlapped call, and read-only calls only ever overlap other read-only calls (never a write). Independence is decided by `concurrency.is_parallel_safe`, deliberately conservative — a false negative only costs a missed speed-up, a false positive could corrupt shared state, so every rule errs toward sequential: `read_file`/`fetch_web` qualify for any args; `bash` qualifies only when `is_read_only_command` proves its `command` is a `|`-pipeline of allow-listed read-only commands with **no** redirection/chaining/substitution metachar (`> < ; & $( \``); `write_file`, `read_tool_artifact` (makes its own LLM call), `delegate` (a subagent can perform any CRUD mid-run — deferred per #65's recorded limitation), and unknown tools never qualify. The worker pool is bounded by `MCA_TOOL_MAX_CONCURRENCY` (default 4; `1` disables overlap). Only the tool *invocation* (I/O-bound work) runs on worker threads (each under a `contextvars.copy_context()` so `current_session_id`/recorder propagate); argument prep, output offloading/dedup, envelope building, result assembly, and every recorder emit stay on the main thread **in call order**, so the shared tool-artifacts store, session files, and recorder capture state are never touched concurrently and each result still maps to its own `tool_call_id` regardless of finish order. Because the recorder's `before_tool`/`after_tool` share a single `_pending` slot (a strict before→after pairing that overlap would race), the concurrent path skips `before_tool` and hands `after_tool` an explicit `(start_mono, end_mono, started_at)` timing bracket each worker captured, so per-call latency/resources reflect the isolated call, not the group's wall-clock.

`tool_execution.schema` is the single, centrally-configured home for every offload-related size threshold (`ARTIFACT_THRESHOLD`, `MAX_TOOL_OUTPUT_CHARS`, `PAGE_FETCH_MAX_CHARS`, `PREVIEW_TOKEN_BUDGET`/`PREVIEW_MAX_CHARS`) — `tool_registry` reads them from there rather than each package defining its own copy. Because `tool_execution/__init__.py` also imports `ToolRegistry`/`artifact_file_path` from `tool_registry` (to dispatch calls and locate artifact files), a naive two-way top-level import would cycle; `ToolExecutor` and `args.strip_unknown_args` instead import `ToolRegistry` lazily (inside the function/method that needs it, or via `type(self.registry)` once an instance exists) so `tool_registry` can safely import the shared constants from `tool_execution.schema` at module load time.

When a tool output is large enough to offload — `bash` combined stdout+stderr, a `read_file` file, or a `fetch_web` page above `ARTIFACT_THRESHOLD` — the tool returns the `(None, artifact_dict)` contract (file/page content in the `stdout` slot) and `after_tool_call` bounds **each stream independently** (`_offload_streams`): a stream over the preview budget (`PREVIEW_MAX_CHARS`) has its **full content written to its own per-stream file** at `.my_coding_agent/<session>/artifacts/<tool_call_id>.<stream>.txt` (`stream` ∈ `stdout`/`stderr`) at creation time and its envelope field carries only a **bounded preview** — a token-bounded excerpt + inline guidance naming `read_tool_artifact(tool_call_id, query=...)` as the primary access path (bash text tools over the on-disk file remain a secondary path); a stream within budget is inlined. `output` carries the stdout preview/inline and `error` the stderr preview/inline, so a large stderr can no longer flood context. `metadata.preview` is keyed by offloaded stream (`{"stdout": {…}, "stderr": {…}}`, shown/total counts + `full_output_path`). The full raw stream never enters the context window; because each file persists for the run, any later step can query it. The path scheme (and its safe-id guard) lives in one place, `tool_registry.artifact_file_path(session_id, tool_call_id, stream)`, shared by the write side (executor) and the read side (`read_tool_artifact`) so the two can never drift. The on-disk write is fault-tolerant: if `mkdir`/`write_text` fails (full disk, permissions), `_write_artifact_file` logs a warning and returns `None` instead of aborting the run — offloading and the preview continue without an on-disk copy, and retrieval falls back to the in-memory store. The preview excerpt is built by `output.build_stream_preview`, applied per stream.

Before writing that per-stream file, `_offload_stream` checks the offload-bound text against this run's already-stored artifacts (`_find_duplicate`, scanning `.my_coding_agent/<session>/artifacts/*.txt` newest-first — deterministic, no LLM): an exact `sha256` hash match, or failing that, containment (the text is a contiguous substring of a stored artifact of equal-or-larger size — needed because `bash` rstrips its own streams, so a read-back of a file with trailing whitespace differs from the original by exactly that). A hit writes **no new file and no preview**; the envelope field instead carries a short pointer naming the original `tool_call_id`/stream/offset/length, and `metadata.duplicate_of` carries the same as a structured, per-stream descriptor (`{"stdout": {tool_call_id, stream, offset, length}, ...}`) — absent when nothing deduplicated. Streams are independent: a duplicate stdout with a novel stderr dedups the former and offloads the latter normally. This closes the read-back loop where the model re-reads an artifact file (e.g. via `bash cat`) and the bytes get offloaded again as a new artifact (issue #92).

### `pipeline/` — DAG Building and Execution

The node-based DAG execution engine. `pipeline/` only knows how to build and execute a DAG — it has no knowledge of LLM client internals or session management.

**`RunContext` (`context.py`)** — the explicit data contract that flows through the pipeline. Holds immutable run config (session id, max steps, LLM client, recorder, all tools) and mutable state fields (messages, step_num, last_prompt_tokens, tool_records, tool_artifacts, handoff_records). Control signals (`signal`, `stop_reason`) are written by nodes and read by `Pipeline.execute`.

**`Node` protocol + `BaseNode` (`node.py`)** — a `Node` is any callable with a `name: str` and a `run(ctx: RunContext) -> None` method. Nodes read and write `ctx` in place.

**`Pipeline` (`dag.py`)** — takes an ordered list of `Node` objects. `run_step` executes every node in order for one step, short-circuiting when any node sets a non-`CONTINUE` signal. `execute` wraps `run_step` in the outer step loop.

**The six default nodes** (instantiated by `build_default_pipeline()`):

| Node | Stage | What it does |
|---|---|---|
| `ContextGuardNode` | 1 | Checks `last_prompt_tokens / context_window`; sets STOP (limit), RESET (handoff), or CONTINUE |
| `ToolRoutingNode` | 2 | Creates `ToolRouter(ctx.llm)`; builds a recency-split work-state signal (`_routing_signal`) — before any assistant message exists, the last user message only; once an assistant message exists, only its content + tool-call names + subsequent tool-result excerpts (capped by `ROUTING_EXCERPT_CHARS`/`ROUTING_SIGNAL_MAX_CHARS`), so the opening task text drops out of the signal entirely once work begins. Calls `route_tools` every step (zero cost) and merges the result with per-run debounce state (`_resolve_selection`): a tag match is an immediate entry, but a previously selected non-baseline tool only exits after two consecutive routings with no tag match (`EXIT_AFTER_MISSES`), absorbing single-step flaps; a mid-run no-match (`phase == "carry_forward"`) decays existing selections without an LLM call. A router event and log line are recorded only when the resulting selection's tool names differ from the previously recorded set (first routing always records) — `events.jsonl` holds one entry per actual decision, not per step |
| `LLMCallNode` | 3 | Increments `step_num`, calls `chat_completion`, appends assistant message |
| `ToolDispatchNode` | 4 | Builds `ToolExecutor(last_message, ctx.llm)`, runs it, merges records and artifacts into `ctx` |
| `AnomalyDetectNode` | 5 | Scans `ctx.tool_records` for a same-signature failure streak (`pipeline/anomaly.py`); at the 3rd consecutive same-class failure (and each further one), records an `anomaly` event and logs one WARNING at first signal. Reads `ctx.tool_records`/`ctx.last_response` only; never touches `ctx.messages`/`ctx.signal`/`ctx.stop_reason`, makes no LLM calls — detection-only, covering subagents for free since every delegated run builds its own pipeline instance |
| `FinalizeStepNode` | 6 | Reads `ctx.last_response`: records token usage + updates `ctx.last_prompt_tokens`, then sets STOP on a stop/exit/quit finish_reason (emits both `token_tracking` and `finish_check` records) |

### `AgentNode` (`engine/agent.py`)

The top-level entry point. Holds an `LLM` client via composition (`self.llm`) — not a subclass. `__init__` builds the client, assigns a session id, attaches the session log, and initializes run stats.

- **`execute(max_steps)`** — stand-alone entry: constructs a `RunContext`, builds the pipeline via `build_default_pipeline(spawn_fn=...)`, delegates to `pipeline.execute(ctx)`, saves session data, and prints the summary. `max_steps` defaults to the shared `DEFAULT_MAX_STEPS` (50) — the single source of truth used by the CLI, the `execute` default, and delegated subagents so all three share one step ceiling.
- **`run(ctx)`** — embedded entry: runs `execute()` and writes results back to the provided `RunContext`; used when `AgentNode` is a step in a larger outer pipeline.

`AgentNode` owns session bookkeeping: banner printing, session log attachment/detachment, `session_data.json` + `tool_artifacts.json` persistence, run summary, and observability recorder start/finish. It also owns `_generate_handoff`, `_spawn_continuation`, and `_handle_context_reset` — the context-reset machinery called by `ContextGuardNode` via the `spawn_fn` closure. A shared `summarize_conversation(llm, messages, prompt, kind)` module-level helper (`pipeline/nodes/context_summarizer.py`) backs both `_generate_handoff` and `generate_report` — a single tool-free LLM call over the whole conversation; when the model returns empty `content` (reasoning models such as Qwen3-thinking often end the summary turn with a tool call, leaving `content` null and the substance in `reasoning_content`), it falls back to `reasoning_content` so the summary is never lost.

A delegated subagent's end-of-turn report is produced by exactly one of three cost-divergent paths, chosen in `delegate()` (`engine/tool_registry/registry.py`), in this order: (1) **verbatim** — a clean finish with a non-empty final assistant turn hands that turn back as-is, zero extra LLM calls; (2) **summarizer** — a step-ceiling cutoff hands back `ctx.handback_report`, already synthesized in-pipeline by `ContextSummarizerNode` as one `report`-kind LLM call; (3) **fallback** — `generate_report()` is the out-of-pipeline last resort (aborted runs, or a clean finish whose final turn carries no usable text), issuing its own `report`-kind LLM call; because `execute()` already saved `session_data.json` before this call runs, `delegate()` re-saves the child afterward so its persisted `total_usage` includes the report's tokens. Every path records its identity as `source` (`verbatim` / `summarizer` / `fallback`, from `engine/schema.py`) on the `report` event via `Recorder.record_report(content, source)` — a required parameter, so a report can never be recorded without declaring which path produced it — making each report's cost statable from `events.jsonl` alone (report-cost-provenance).

`AgentNode` also accumulates a **rollup**: `child_rollups` collects each delegated subagent's usage summary (own per-kind totals, elapsed seconds, step count, its own nested descendants, and its report's `source`) as `delegate()` hands it up via the `current_agent_node` contextvar — write-side, no re-reading of child files. `_usage_summary()` combines this agent's own per-kind totals with `child_rollups` into a `grand_total` (own + every descendant, transitively), persisted as `session_data.json`'s `rollup` key and surfaced in the terminal run summary as a task-level cost section (own vs. rolled-up totals, one line per direct subagent marked free/paid by its report `source`) whenever delegations occurred — unchanged output otherwise. `viewer/sumcheck.py` provides a deterministic, LLM-free check over a session directory tree: per-call usage rows sum to the persisted `by_kind`, own totals plus descendants' grand totals equal `grand_total`, and a report's `source` is consistent with the presence/absence of a `report`-kind usage row — exposed via `my-coding-agent-traces --check <session_id>` (exit 0/1; incomplete records report as unverifiable, not failed).

### Context Handoff (`pipeline/schema.py` + `pipeline/handoff.py` + `pipeline/nodes/context_guard.py`)

`pipeline/schema.py` holds the `ContextHandoff` dataclass shape only; `pipeline/handoff.py` holds the builder logic that operates on it — `save_handoff()` (persistence) and `handoff_to_user_message()` (rendering) — kept out of `schema.py` per the Schema Convention.

When `ContextGuardNode` detects that `prompt_tokens / context_window >= context_reset_threshold`, it calls `spawn_fn()` — a closure set by `AgentNode.execute` that calls `AgentNode._handle_context_reset`. That method:
1. Sends a structured handoff prompt asking the LLM to summarize progress, files changed, decisions made, and remaining work.
2. Saves the result as a markdown file under `.my_coding_agent/handoffs/` via `save_handoff()`.
3. Saves session data and prints a summary for the current run.
4. Spawns a fresh `AgentNode` instance carrying only the system messages and the handoff as a user message.

`ContextGuardNode` stores the result in `ctx.continuation_messages` and sets `ctx.signal = "RESET"`, which causes `Pipeline.execute` to return those messages immediately.

### `ToolRegistry` (`engine/tool_registry/` package)

A plain class whose methods are the tools the LLM can call:

| Tool | Purpose |
|---|---|
| `bash(command, timeout, stdin)` | Runs a shell command; reports `stdout, stderr, exit_code, ok`; large combined output offloads. `stdin` delivers text to the command's standard input directly (subprocess `input=`), bypassing shell composition — the structural channel for multi-line scripts (written via `write_file`, run with `bash`, data via `stdin`) instead of pipe+heredoc or quote-nested `-c` one-liners |
| `read_file(file_path)` | Reads a file; files over `ARTIFACT_THRESHOLD` offload (lossless — the full content is stored, not truncated) |
| `write_file(file_path, content)` | Writes a file, creating parent dirs |
| `fetch_web(url)` | Fetches any text URL; converts to markdown only for HTML responses (`text/html`, `application/xhtml+xml`) — every other text type (JSON, plain text, XML, …) passes through verbatim, and non-text types (`image/*`, `application/octet-stream`, …) are rejected with an explicit error. Every result's `metadata` discloses the served `content_type` and `transform` (`html-to-markdown` or `none`). Bodies over `ARTIFACT_THRESHOLD` offload (lossless within a generous fetch-side sanity cap, `PAGE_FETCH_MAX_CHARS`; on the verbatim path, cap truncation is signaled via `metadata.truncated`, never appended into the body) |
| `read_tool_artifact(tool_call_id, query, start, length)` | Two mutually exclusive modes. Query mode (default, `query` required): makes a bounded local-LLM extraction call (`artifact_query` kind) over the stored content and returns only a token-bounded extract relevant to `query` — never the full stored content, regardless of query, artifact size, or failure mode. Artifacts over the per-call input budget are scanned in sequential chunks so any part of the output is reachable, not just a prefix. Degrades to a bounded head excerpt + on-disk path hint when no LLM client is injected or the extraction call fails. Any incompleteness in query mode — a chunk completion cut at the extraction token cap, the joined result sliced to the output budget, or chunks left unscanned when the budget fills — is disclosed as a trailing bracketed marker in the returned text naming the cause, the scanned-vs-total extent, and a recovery path (narrower re-query, byte-range retrieval, or bash-skim of the on-disk file); a complete extract carries no marker. Cap detection tolerates a provider that omits `finish_reason` on a capped completion (`finish_reason == "length"`, or an empty reason with `completion_tokens` at the budget). The completion cap in force for an extraction call is also recorded on its `llm_call` trace event so the viewer can badge a cut `artifact_query` call without importing the engine's budget constant. Range mode (`start` given, `query` ignored): returns an exact, verbatim byte-range slice — `start`-based, `length` defaulting to and capped at `RANGE_MAX_CHARS` (aliased to `PREVIEW_MAX_CHARS`) — prefixed by a one-line range/total header, no LLM call; the only mode that works on content with little or no line structure, and what a `duplicate_of` pointer's offset/length feed into. An unknown `tool_call_id` or out-of-range `start` returns a clear error (the latter naming the total size) |
| `delegate(task, known_facts="")` | Spawns a fresh read-only subagent for a focused task; the subagent inherits the parent toolset **minus `delegate`** (to prevent recursive spawning) and runs with the same `DEFAULT_MAX_STEPS` budget as the main agent. `known_facts` is optional and additive-only — facts the task text doesn't already state (file paths, environment facts, constraints); its schema description tells the parent never to restate the task there, and a deterministic per-line lexical-containment guard (`strip_task_restatements`) drops any line that duplicates or near-duplicates the task before the child's opening message is built, so a restatement that slips past the schema guidance never reaches the child's prefix. The child's opening message is task-only when there's nothing additive, or `"Task:\n{task}\n\nKnown facts from the main agent:\n{facts}"` otherwise. Returns the subagent's report by cost order: a clean finish's final assistant turn verbatim (zero extra LLM calls); otherwise the in-pipeline synthesized `handback_report` (one `report`-kind call already made); otherwise `generate_report()`'s out-of-pipeline fallback call — never a stale scrape of an earlier message, so the final tool results survive even when the subagent is cut off at its step ceiling. Each path's identity is recorded as the report event's `source`, so which path fired (and whether it cost an LLM call) is auditable from the trace alone |

The `@tool` decorator converts any `ToolRegistry` method into an OpenAI-compatible tool definition by inspecting its signature and parsing Google-style docstrings.

### `observability/` — Passive Event Capture

Receives events emitted by `engine/` and `pipeline/`; never controls execution. Writing directly to `events.jsonl` is its only side-effect.

- **`sampler.py`** — `ResourceSampler`, a background daemon thread sampling machine-wide RAM/CPU/network/disk (`psutil`, no privileges) and Apple Silicon GPU utilization (`ioreg -r -d 1 -c IOAccelerator`, parsing `PerformanceStatistics` → `Device Utilization %` — the only sudo-free GPU signal on macOS; `powermetrics` needs root) into a bounded in-memory ring buffer at a ~1 s interval. `summarize_window(start, end)` returns gauges (RAM/CPU/GPU) as avg+peak and counters (network/disk) as edge deltas over a monotonic window, falling back to one on-demand snapshot when the window is shorter than the sampling interval; `session_rollup()` returns the same shapes over the whole run, folding samples evicted from the ring into a running total so long runs stay correct. Any sampling or snapshot exception disables further capture and logs one warning — never raises into the run. `get_sampler()` returns one process-wide singleton; `start`/`stop` are reference-counted so the main agent and in-process subagents (sharing identical machine-wide data) don't tear the thread down under each other.
- **`recorder.py`** — event type constants (`SESSION_START`, `LLM_CALL`, `TOKEN_TRACKING`, `FINISH_CHECK`, `HANDOFF`, `REPORT`, `ANOMALY`, etc.); `Recorder` appends events as newline-delimited JSON. `Recorder.start`/`finish` start/stop the shared `ResourceSampler`; `resource_window(start, end)` and `resource_rollup()` are thin pass-throughs so call sites don't import the sampler directly. `record_llm_call` and `record_summarizer` accept an optional `resources` object (the caller's sampler-window summary for the call's bracket) written onto the event only when not `None`; `after_tool` computes its own window from the pending tool-start time it already tracks internally — except on the concurrent dispatch path (issue #65), where the executor passes an explicit `(start_mono, end_mono, started_at)` bracket via `timing=` and the shared `_pending` slot is left untouched so overlapping calls can't race it. `record_anomaly(kind, streak_id, signature, tool_name, streak_len, tokens_spent, step)` is purely passive — it records what `AnomalyDetectNode` reports and never judges or influences execution; rows sharing one `streak_id` let consumers keep only the latest (final) magnitude for a streak. Every timed event (`llm_call`, `tool_call`, `summarizer`) records a **true start timestamp**: the caller captures wall-clock time alongside its monotonic latency timer (`time.monotonic()`) and passes it through as `started_at`, rather than the recorder deriving it from `_now()` at emit time — so `started_at + latency_s` fits within the session's own bounds instead of the stamp mislabeling completion time as start. `_now()` (the emit-time fallback, and the source for untimed events) formats ISO-8601 with millisecond precision and a UTC offset (`datetime.now().astimezone().isoformat(timespec="milliseconds")`); the reader tolerates the pre-change format (second precision, naive) since it never parses timestamps, only stores them. `record_llm_call` captures each call's full input — the `messages` snapshot and the `tools` definitions given to the model that turn — for kinds in `FULL_PAYLOAD_KINDS` (both dropped otherwise to bound stream size). To keep `events.jsonl` growing linearly instead of quadratically with conversation length, the `messages` snapshot itself is delta-encoded per call kind (`_encode_messages`, incremental-trace-capture D1/D2): the recorder keeps, per kind, the last emitted snapshot's physical call number and message-object references; on the next call of that kind, if the current messages provably extend that snapshot's prefix (verified by `id()` identity over the shared region — never by content comparison, and never mutated in place per the append-or-replace invariant on `ctx.messages`), the event carries `messages: null` plus `messages_base_call`/`messages_prefix_len`/`messages_suffix` instead of the full list. Any identity mismatch, a shorter list, or the first call of a kind falls back to a full snapshot — fidelity always wins over size. `record_report(content, source)` captures a subagent's end-of-turn final report as a `REPORT` event, distinct from the context-reset `HANDOFF` event; `source` (required, from `engine/schema.py`'s `REPORT_SOURCE_*` constants) names which of the three cost-divergent paths produced it. `after_tool(name, args, result, ok, error, timing=None)` requires the outcome the executor already holds (never re-derived by parsing `result`): every `tool_call` event carries a top-level `ok`, and on failure (`ok: false`) an `error` text plus an `error_class` computed by the shared helper `observability.classify_error` (the same normalization `pipeline/anomaly.error_signature` delegates to, so the two classifications agree by construction) — a consumer answers "which calls failed, with what class" with one `json.loads` per line, no parsing of `result`. Pre-change traces simply lack these fields; the reader falls back to parsing `result` only for those. Three `ContextVar`s (`current_session_id`, `current_recorder`, `current_agent_node`) let delegated subagents record their parent link and hand usage totals up to the parent `AgentNode` without re-reading child files.

### `viewer/` — Active Read-Side (Trace Explorer)

The read-side of the observability system. Separated from `observability/` because it is **active** — it controls execution (HTTP server), renders output (embedded HTML), and manages file handles — whereas `observability/` is passive capture only (CONTRIBUTE.md §25).

- **`schema.py`** — `TraceNode` and `TraceSession` dataclasses: the typed contracts produced by `reader.py` and consumed by `server.py`. Machine-wide `resources` (when present) rides in `TraceNode.attributes["resources"]` for `llm_call`/`tool_call`/`summarizer` nodes — no dedicated field, since `attributes` is already the free-form scalar-metadata bag; the run-level rollup rides in `TraceSession.analytics["resource_rollup"]`. `TraceNode.ctx_state` holds the per-node context-window snapshot — cumulative `composition` by `system`/`user`/`assistant`/`tool` role, the per-role `added` this node appended (with `added_total`/`removed` and an `estimated` flag), plus `tokens`/`window`/`pct`. Each node also carries `agent` (owning session id) and `depth` (call-tree nesting level), so sub-agent traces nest under the main agent. `TraceSession.order` is the execution-order node spine the UI walks for keyboard navigation.
- **`pricing.py`** — model price table (USD per 1M tokens) and `compute_cost()` helper.
- **`reader.py`** — `_read_events` resolves any prefix-delta-encoded `llm_call` events (`_resolve_message_deltas`) before anything else touches the parsed rows: it walks events in file order, keyed by physical call number, reconstructing `base[:prefix_len] + suffix` (chaining through a base that is itself a delta) so every downstream consumer sees the exact full `messages` snapshot the call saw — byte-identical to full-snapshot capture, both for legacy traces (detected by `messages` already non-null) and new delta traces. A delta whose base call number is absent from the file (corrupt/hand-truncated mid-event) degrades to that event's `messages` being `None` rather than failing the load; any prefix of a crash-truncated file still reconstructs every event fully written before the truncation, since a delta only ever references an earlier line in the same file. Parses `events.jsonl` into a **flat** `TraceSession`: every pipeline `BaseNode` subclass (`ContextGuardNode`, `ToolRoutingNode`, `LLMCallNode`, `ToolDispatchNode`, `FinalizeStepNode`) becomes one `TraceNode` in a single chain off the session root — there is no `step` wrapper node; the step number is carried as an attribute. A subagent's `REPORT` event becomes a distinct `report` node (`Subagent Report`), separate from the context-reset `handoff` node; its `source` attribute (verbatim/summarizer/fallback) surfaces as `"unknown"` when absent (pre-provenance traces), never an error or a guessed path. Reconstructs the context window as four role buckets in execution order (`_assign_ctx_state`): each node contributes the message(s) it appends — the session seeds system + opening user, an LLM call adds its `assistant` output (exact `completion`), a tool dispatch adds its result (character-estimated via a session tokens/char ratio, since tool tokens are never recorded); composition re-anchors to each LLM call's real input snapshot (`_role_split` splits the provider's flat `prompt_tokens` across the four roles by character share). Each agent gets its **own** context window: `_assign_ctx_state` processes only its session's nodes, so delegate sub-agents (loaded recursively at an incremented `depth` and grafted inline) keep the independent windows computed by their own load. `_compute_analytics` sums token totals across **every** call kind (not just `main`) across the grafted parent+children graph, and adds `by_kind` (tokens per call kind) and `by_agent` (per session id: tokens, call count, and wall time read off that agent's own `session_end.elapsed_s`) — `llm_call_count` still counts `main` calls only, for step semantics. `_read_resource_rollup` reads `resource_rollup` straight out of `session_data.json` in the same session directory (written once the run finishes) and, when present, folds it into `analytics["resource_rollup"]`; missing file or missing key both resolve to `None` without error, so a session without resource capture (or still in progress) surfaces no rollup. Timestamps are never parsed (`started_at`/`ended_at` stay opaque strings), so pre-change traces and crash-truncated traces (missing `session_end`, or missing `session_data.json` entirely — falls back to a two-node stub) load without error, with analytics computed purely from the event rows present. Also does loop detection. `anomaly` rows become one `anomaly` trace node per `streak_id` (`_add_anomaly_nodes`, keeping only the latest/final-magnitude row per streak), and `_flag_anomalies` sets `anomaly_flag` on the streak's own failing `tool_call` nodes via a positional back-walk (matching tool name + a failed result — read from the node's top-level `ok` attribute when the event recorded one, else parsed from the raw JSON `result` for pre-change traces; `tool_call` events carry no `tool_call_id` to link exactly), separate from and alongside the existing `loop_flag`.
- **`server.py`** — minimal stdlib `http.server` with three routes (`/`, `/api/sessions`, `/api/session/{id}`) and an embedded single-page Trace Explorer UI built with **Preact + htm** (vendored offline under `_vendor/`, injected inline — no CDN, no build step). The UI is a nested call-**Tree** (Main Agent at the root; each `delegate` spawns a collapsible **Subagent** group, with a coloured rail, nested where it was called — derived from each node's `agent`/`depth`) with keyboard navigation (auto-select), a type filter, and a single per-node **Context window** box (a system/user/assistant/tool composition bar + legend and the running total/%, badged with the owning agent for sub-agents since each tracks its own window); Each tree row shows the node's compact badges then its contribution summary (e.g. *+196 assistant*) — including grouped rows (any node whose dispatch nests children, e.g. a `delegate` tool_call nesting its subagent session root), which carry their own ctx-window contribution just like leaf rows and must render it too. Node detail leads with a **uniform header** shared by every node type — a single big **main badge** naming the node class (e.g. `ToolDispatchNode`, the `(tool)` suffix stripped since it appears as its own badge) and a badge row ordered by importance left→right: identity (tool name / llm kind / model) · status (✓/✗, colored) · colored type signals (📦 artifact, ✂️ truncated, ✂️ cut at N-token cap for a capped `artifact_query` llm_call, 📜 multi-line for a `bash` call with a non-empty `stdin` or a newline in `command`, 🧭 phase, 🆓/💰/❔ free/paid/unknown for a `report` node's `source`, 🖥 machine-wide cpu/ram/gpu figures for the node's window) · counts (N tools / N calls / N steps) · then de-emphasised housekeeping (⚡ latency, 🕘 timestamp, step); only badges whose data exists are shown, and the same `nodeBadges` set drives the compact tree badges. A flagged `tool_call` node's header also shows a distinct `anomaly` tag next to (never replacing) the existing `loop` tag, and the stats strip shows an anomaly count alongside the loop count when either is present, plus a `machine (run)` breakdown row surfacing the session's resource rollup when captured. Below the header it is type-aware: a tool dispatch renders four labelled boxes — **command**, **output**, **error**, and the always-shown **raw_envelope** (`{schema_version, tool, ok, output, error, metadata}`) — each highlighted in its `metadata.lang` language, and empty output / absent error omitted so the raw envelope still guarantees an empty `output` whose real signal lives in `error` is never hidden. LLM calls render response/reasoning/tool-calls boxes, and a `report` node renders the subagent's final report content. An LLM call's **Inputs** surface both the `messages` snapshot and the `tools` definitions the model was given that turn. Every content box renders through one read-only `CodeBox` component wrapping a **CodeMirror 6** editor (vendored offline): a box takes an explicit language (`json`/`python`/`shell`/`text`, from the backend `metadata.lang` hint) or falls back to JSON-vs-text inference. It provides syntax highlighting, line numbers, folding, a JSON schema breadcrumb derived from the caret's syntax-tree path (clickable to jump; JSON only), collapse-all/expand-all, copy-all, and `@codemirror/search` find (open + Enter/Shift+Enter next/prev). Editors are created only for the selected node's visible boxes, so many-node sessions stay fast. CLI entry point: `my-coding-agent-traces [--port 7474] [--dir .my_coding_agent]`, or `--check <session_id>` to run the deterministic sum-check (`viewer/sumcheck.py`) instead of starting the server, exiting 0/1.
- **`_vendor/`** — third-party UI libraries (Preact, Preact Hooks, htm as UMD bundles; CodeMirror 6 as a prebuilt IIFE bundle exposing `window.CM6` with `json`/`python`/`shell` languages — shell via `@codemirror/legacy-modes`, no `lang-shell` package exists) vendored offline so the localhost viewer needs no internet. JS only; excluded from coverage.

### `utils/` — Generic Helpers

Active utilities that configure loggers and render to stderr — not passive, so they live outside `observability/`.

- **`logging_core.py`** — custom log levels `TOOL` (15), `API` (25), `LLM` (35); `ColoredFormatter`; `get_logger`; `_TeeStream` + `attach_session_log` / `detach_session_log` that tee stderr to per-session log files.
- **`terminal_ui.py`** — `print_banner` (startup box) and `print_run_summary` (end-of-run box with token chart); shared `_git_branch` helper; all row/section/chart sub-helpers, including `_resource_rollup_section` (machine-wide RAM/CPU/GPU/network/disk, rendered only when the session captured resource data). Both renderers write directly to `sys.stderr`, bypassing the logger formatter.

### Schema Convention

Every module and sub-module owns a `schema.py` for its typed contracts and shape definitions. Builder logic stays in its own module; `schema.py` holds constants, type aliases, and dataclasses only.

| Module | `schema.py` contents |
|---|---|
| `engine/schema.py` | Session/LLM/tool/handoff/report event type constants |
| `engine/llm/schema.py` | LLM call kind constants (`CALL_KIND_*`, sole source of truth — consumed by `agent.py`, `routing.py`, and every `pipeline/nodes/*.py` call site instead of raw string literals), usage field names |
| `engine/tool_execution/schema.py` | Canonical tool-result envelope shapes (builders live in `envelope.py`), plus every size/threshold constant that decides artifact offloading (`ARTIFACT_THRESHOLD`, `MAX_TOOL_OUTPUT_CHARS`, `PAGE_FETCH_MAX_CHARS`, `PREVIEW_TOKEN_BUDGET`/`PREVIEW_MAX_CHARS`) — a leaf module (no internal imports) so both `tool_execution` and `tool_registry` read it without a cycle |
| `engine/tool_execution/concurrency.py` | The read-only independence gate for concurrent dispatch (issue #65): `is_parallel_safe(func_name, args)` (conservative, errs toward sequential), `is_read_only_command` (allow-listed `bash` pipelines with no write/chaining/substitution metachars), and `max_tool_concurrency()` (bounded worker ceiling from `MCA_TOOL_MAX_CONCURRENCY`, default 4, `1` disables). Leaf module (stdlib only) |
| `engine/tool_registry/schema.py` | OpenAI tool definition JSON key names |
| `pipeline/schema.py` | ROUTER event type constant, `ContextHandoff` dataclass (builders live in `handoff.py`) |
| `observability/schema.py` | JSONL row top-level key names; machine-wide `resources` object field names (`RESOURCES_*`) |
| `viewer/schema.py` | `TraceNode` + `TraceSession` dataclasses |

---

## Entry Point (`pipeline/examples/simple.py`)

```
CLI (Click)
  │
  └── AgentNode (engine/agent.py)
        System prompt: stable core (identity, tool-usage/safety rules, envelope
          contract, environment facts) + trailing timestamp — no tool docs, no
          repo state (tools reach the model only via the structured schema)
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
| `session_data.json` | Metrics, tool records, LLM call log, stop reason, usage `rollup` (own per-kind totals + per-descendant subagent entries + grand total), and — when resource capture was available — `resource_rollup` (session-wide machine-wide RAM/CPU/GPU peaks/averages, network/disk byte totals) |
| `events.jsonl` | Structured observability event stream; timed events (`llm_call`, `tool_call`, `summarizer`) may carry a `resources` object (machine-wide load over the event's window) when capture was available |
| `artifacts/<tool_call_id>.<stream>.txt` | Full content of each offloaded large output stream (`stdout`/`stderr`), written at creation so bash can skim it during the run |
| `tool_artifacts.json` | End-of-run audit dump of the in-memory artifact records |

Handoffs are saved under `.my_coding_agent/handoffs/`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | OpenAI-compatible API base URL |
| `OMLX_API_KEY` | `changeme` | API key |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-6bit` | Model ID |

The context window size is discovered lazily on first access to `LLM.context_window`.

When the `bash` tool's combined stdout/stderr exceeds the artifact threshold, the dispatcher offloads it to a per-stream file and substitutes a deterministic bounded preview — it does not generate an LLM summary.

The Trace Explorer's machine-wide resource badge (`🖥 cpu/ram/gpu`, node-resource-monitoring) renders only in the node detail panel, not the tree rows — it carries its own `res` badge class (distinct from `lat`) so `TREE_BADGE` can exclude it, keeping the tree scannable.


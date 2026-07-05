# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Concurrent tool dispatch (issue #65)

`ToolExecutor` (`engine/tool_execution/__init__.py`) may **overlap** the tool calls in one assistant message, but only within strict safety rails — read them before touching this path:

- **Only provably read-only-per-call calls overlap**, and only with each other. `concurrency.is_parallel_safe` is the single gate and is deliberately conservative: a false negative just misses a speed-up, a false positive can corrupt shared state. `read_file`/`fetch_web` qualify for any args; `bash` only when `is_read_only_command` proves it (allow-listed `|`-pipeline, no `> < ; & $( \`` metachars). `write_file`, `read_tool_artifact` (makes its own LLM call), `delegate` (a subagent can do any CRUD mid-run — deferred), and unknown tools never qualify. **Do not add a tool to the allow-list without proving it cannot write local or remote state under any argument set** (e.g. `sort -o`, `uniq out`, `tree -o`, `env VAR=x cmd`, `curl -o`/`-X POST` are all write-capable and excluded on purpose).
- **A non-overlappable call is a barrier** (`_plan_groups`): ordering across groups is preserved exactly like the old sequential path, so results always map to their `tool_call_id` regardless of finish order.
- **The shared mutable process state must never be touched concurrently.** Only the tool *invocation* runs on worker threads; arg prep, artifact-store writes/dedup, envelope building, result assembly, and every recorder emit stay on the main thread in call order. The `Recorder`'s `before_tool`/`after_tool` share a single `_pending` slot (a strict before→after pairing) — the concurrent path skips `before_tool` and passes `after_tool` an explicit `timing=(start_mono, end_mono, started_at)` bracket instead, leaving `_pending`/`_pending_child_llm_calls` untouched. If you make more of the dispatch path concurrent, keep these off the worker threads or the trace/session files will corrupt.
- Bounded by `MCA_TOOL_MAX_CONCURRENCY` (default 4; `1` disables overlap). Stdlib only (`concurrent.futures.ThreadPoolExecutor`, `contextvars.copy_context`).

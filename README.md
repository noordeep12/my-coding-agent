# my-coding-agent

A personal, from-scratch Python agent framework for running LLM-powered coding agents against local LLM servers. Built for me.

## What It Is

A self-contained agentic loop — no external agent frameworks. The agent calls a local model, executes tool calls, feeds results back, and repeats until the task is done or a step/context limit is hit.

Key ideas:
- **Local-first**: targets OpenAI-compatible endpoints (MLX Server, Ollama) running on your machine
- **Node-based pipeline**: the agentic loop is a DAG of named nodes (`ContextGuard → ToolRouting → LLMCall → ToolDispatch → AnomalyDetect → FinalizeStep`) with an explicit data contract (`RunContext`) flowing between them
- **Runtime anomaly detection**: while the run is live, a same-class tool-failure streak (e.g. the same tool failing 3+ times in a row with the same error class, regardless of args) is flagged the moment it happens — logged as a warning and recorded in the session's event stream, in main agents and subagents alike
- **Decorator-based tools**: plain Python functions become LLM-callable tools
- **Skills**: steer the agent's tool usage with plain-Markdown `SKILL.md` files, loaded on demand via `use_skill` — no Python edits, no system-prompt changes
- **Context handoff**: when the context window fills up, the agent writes a structured summary of progress and spawns a fresh continuation — so long-running tasks don't get silently truncated
- **Run resilience & resume**: a mid-run LLM failure is classified at the client and, when transient (server restart, model reload, Mac sleep), absorbed by a patient bounded retry; each completed step is checkpointed, so a dead run can be continued from exactly where it left off with `--resume`
- **Session persistence**: each run saves token usage, tool calls, and a final summary to `.my_coding_agent/<session_id>/`

## Quick Start

```bash
# Start a local LLM server
mlx-llm server --model Qwen3.6-35B-A3B-6bit
# or
ollama serve

# Install dependencies and run
uv sync
uv run my-coding-agent
```

## Installing & Updating Locally

This is a [src-layout](https://hatch.pypa.io/latest/config/build/#packages) package managed with `uv`.

```bash
# Install runtime deps into a local .venv (Python 3.12+ is required)
uv sync

# Include the dev tools (ruff, mypy, pytest, sphinx, ...)
uv sync --all-groups
```

`uv run my-coding-agent` runs the package straight from `src/`, so **any code change is picked up on the next run with no reinstall step** — there is no separate `pip install -e .` to repeat. Re-run `uv sync` only when dependencies in `pyproject.toml` change.

## Entrypoint

The `my-coding-agent` console script is declared in `pyproject.toml`:

```toml
[project.scripts]
my-coding-agent = "my_coding_agent.pipeline.examples.simple:main"
```

It points to `main()` in `src/my_coding_agent/pipeline/examples/simple.py`. To change the default command, edit that mapping (and run `uv sync` to refresh the script); to change the default behaviour, edit the Click options on `main()`.

## CLI Usage

```bash
# Default: run with the default commit-and-push task
uv run my-coding-agent

# Custom prompt:
uv run my-coding-agent --prompt "Your task here"

# Interactive paste mode (Esc then Enter, or Meta/Alt+Enter to submit; Ctrl+C to cancel)
uv run my-coding-agent --interactive

# Resume a dead run from its last checkpoint (new session, linked to the dead one)
uv run my-coding-agent --resume <session_id>
uv run my-coding-agent --resume-last

# Help
uv run my-coding-agent --help
```

### Run Resilience & Resume

Local model time is the run's real cost, so a transient LLM failure must not throw it away. Two mechanisms cover the two faces of that:

- **Live resilience.** Every failure is classified at the LLM client (the single choke point every call kind funnels through): *transport* (connection/timeout), *http-status* (non-2xx), or *malformed-body* (non-JSON, or JSON with missing/empty `choices`). A failed response never becomes an empty assistant turn. Retryable classes (transport, HTTP 5xx/429) get the existing fast retries and then a **patient bounded wait** — capped-backoff probing until the server answers or `MCA_LLM_OUTAGE_TOLERANCE_S` (default 300s) is exhausted — so a server restart, model swap, or brief Mac sleep is absorbed unattended. Non-retryable classes (other 4xx) fail immediately. Every wait, recovery, and unrecoverable failure is logged (`stderr.log`) and recorded (`events.jsonl`).
- **Resume.** After each *completed* step the engine writes `checkpoint.json` (the exact conversation plus step/token counters) atomically. If a run dies — outage beyond tolerance, `kill -9`, crash — `--resume <session_id>` (or `--resume-last`) loads that checkpoint into a **new** session linked back to the dead one (`resumed_from`); the dead session's files stay immutable and the resumed run continues from step N+1, not step 0.

**Partial-step caveat (by design):** the checkpoint granularity is a completed step. A run killed mid-step resumes from the end of the last *completed* step; the partial step is discarded. Any tool side effects from that partial step may already exist on disk — the resumed conversation simply doesn't know about them (the same risk class as re-running a command after a crash). There is no step-level journaling.

**Known limitations (by design):**

- **Multi-hop resume chains.** The automatic source-checkpoint cleanup clears only the *immediate* source, and only on a clean finish; a `max_steps` hop deliberately keeps its checkpoint (still resumable with a larger budget). So a completed resume chain of 3+ hops that includes a `max_steps` middle hop (e.g. `W` dies → `X` resumes `W` and hits `max_steps` → `Z` resumes `X` and finishes) can leave an earlier ancestor's checkpoint (`W`) behind, which `--resume-last` may then target and re-run an already-finished task. No progress is lost and the conversation is intact. Workaround: pass `--resume <id>` explicitly, or delete the stale `.my_coding_agent/<id>/checkpoint.json`.
- **Propagated continuation-failure trace artifacts.** When a post-context-reset continuation fails unrecoverably and the failure is propagated to the main run, the main session's `session_data.json` (written at reset time with `stop_reason=context_reset`) and its `session_end` event (`stop_reason=llm_failure_*`) can disagree — a cosmetic trace-observability inconsistency only. The authoritative failure and resume state lives in the continuation session, which holds the resumable checkpoint and is named by the CLI resume hint.

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--prompt`, `-p` | default commit-and-push task | Task for the agent |
| `--interactive`, `-i` | `False` | Read the task prompt interactively (Esc then Enter, or Meta/Alt+Enter to submit; Ctrl+C to cancel) |
| `--max-steps` | `50` | Maximum pipeline step iterations |
| `--resume` | — | Resume a dead session from its last checkpoint (`--resume <session_id>`); starts a new session linked back to the dead one |
| `--resume-last` | `False` | Resume the most recently checkpointed session |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-6bit` | Model ID to use |
| `MCA_TOOL_MAX_CONCURRENCY` | `4` | Max read-only tool calls overlapped per assistant message (`1` disables overlap) |
| `MCA_LLM_OUTAGE_TOLERANCE_S` | `300` | Seconds the client keeps probing a stalled/restarting LLM server (transport / HTTP 5xx / 429) before giving up with a classified, resumable stop; other 4xx fail fast |
| `MCA_SUPERSESSION` | `1` | Set to `0` to disable retiring provably-superseded tool results (restores append-only conversation behavior byte-for-byte) |


## Skills

Skills let you steer the agent's tool usage with plain Markdown, without editing any Python. A skill is a directory holding a `SKILL.md` file that bundles procedural knowledge for a specific task ("when doing X, drive the tools this way").

**Locations** (scanned once at session start; a project skill shadows a user skill of the same name):

| Scope | Path |
|---|---|
| Project | `<cwd>/.my_coding_agent/skills/<name>/SKILL.md` |
| User | `~/.my_coding_agent/skills/<name>/SKILL.md` |

**Format** — minimal frontmatter (`name` and `description`) plus a free-Markdown body. `description` doubles as the when-to-use hint shown in the index; unknown frontmatter keys are ignored (so Claude Code-authored `SKILL.md` files parse as-is). Malformed frontmatter skips that skill with one warning — it never fails the run.

```markdown
---
name: commit-and-push
description: Commit staged changes and push with a Conventional Commits message.
---

1. Run `git status` and `git diff --staged` to see what changed.
2. Group related changes; write a `type(scope): summary` subject ≤ 72 chars.
3. Commit, then `git push` to the current branch's upstream.
```

**How it reaches the model** — when at least one skill is discovered, a compact index (one `- name: description` line per skill, within a fixed character budget) is appended to the opening task message, and a `use_skill(name)` tool is registered. The system prompt is never touched, so the prompt-prefix cache still hits across runs. The agent calls `use_skill("commit-and-push")` to pull that skill's full instructions into context on demand; a skill is loaded once (a repeat call returns a short pointer). Delegated subagents get the same index and tool; the loaded skills survive a context handoff. With **no** skills on disk, nothing changes — no index, no `use_skill` tool, identical prompts and tool schemas.

**Bundled example skills** — the repo ships three ready-to-use project skills under `.my_coding_agent/skills/`. That subtree is un-ignored in `.gitignore` (the rest of `.my_coding_agent/` stays ephemeral and untracked), so they are committed and **auto-discovered** with no setup — run the agent from the repo root and they appear in the index.

| Skill | When to use |
|---|---|
| `commit-and-push` | Stage, write a Conventional Commit that passes the `commit-msg` hooks, and push the current branch. |
| `run-quality-checks` | Run tests + ruff + mypy + security the way the `Makefile`/CI expect before declaring work done. |
| `debug-failing-run` | Diagnose a failed run from its session files (`events.jsonl`, `session_data.json`) and the Trace Explorer. |

Treat them as templates: copy one into a new `.my_coding_agent/skills/<name>/SKILL.md` (or `~/.my_coding_agent/skills/` for a user-wide skill) and edit the frontmatter and body for your own task.


## Observability

Every run automatically records a structured `events.jsonl` alongside the other session files under `.my_coding_agent/<session_id>/`. No configuration needed — capture is always on.

| File | Contents |
|---|---|
| `stderr.log` | Plain-text log of the full run |
| `session_data.json` | Metrics, tool records, LLM call log, stop reason, usage rollup (own + delegated subagents, per call kind) |
| `events.jsonl` | Structured event stream (LLM calls, tool I/O, handoffs, skill-index offers, outage waits/recovery) |
| `checkpoint.json` | Engine-owned per-step resume checkpoint (exact `messages` + step/token counters), written atomically after each completed step; consumed by `--resume` |
| `artifacts/<tool_call_id>.<stream>.txt` | Full content of each offloaded large output stream (`stdout`/`stderr`), written at creation so bash can skim it during the run |
| `tool_artifacts.json` | End-of-run audit dump of the in-memory artifact records |

### Machine-wide resource capture

Since the LLM runs locally, machine load *is* the run's real cost. A background sampler (no sudo required) records **machine-wide** RAM/CPU/GPU/network/disk figures — what the Mac was doing, not per-process attribution — for the execution window of every timed event (`llm_call`, `tool_call`, `summarizer`), and a session-wide rollup (peaks/averages, byte totals) is persisted in `session_data.json` alongside the token/time totals. These figures are explicitly labeled `machine_wide: true` since other processes on the machine (e.g. a browser) count toward them too. Capture is passive (never throttles or aborts the run) and degrades gracefully: if the GPU reading is unavailable (non-macOS, or the `ioreg` output changes) or the sampler ever fails, the affected field (or all `resources` data) is simply absent — the run and its trace are unaffected. Sessions recorded before this existed load and render unchanged. Both the terminal run summary and the Trace Explorer (node detail, session stats) show these figures whenever they were captured.

### Trace Explorer

Visualise sessions in a browser with an interactive pipeline DAG:

```bash
my-coding-agent-traces          # defaults: port 7474, dir .my_coding_agent
my-coding-agent-traces --port 8080 --dir /path/to/.my_coding_agent
my-coding-agent-traces --dir .my_coding_agent --check <session_id>  # deterministic sum-check, no server; exit 0/1
```

Then open `http://localhost:7474`. The UI (an Apple-minimalist Preact app, served fully offline) shows:

- **Left pane** — a nested **Tree** of the run: the Main Agent's pipeline, with each delegated **subagent** nested (collapsible, with a coloured rail and badge) where it was spawned; each node label summarises what it added to the context window (e.g. *+196 assistant*, *+1,501 tool*)
  - Navigate with **↑/↓ (or j/k)** arrow keys; the focused node auto-selects
  - **Filters** button — show/hide nodes by type
- **Right pane** — node detail:
  - a single **Context window** box: the running window (total / max · %) and a composition bar + legend split into **system / user / assistant / tool** tokens, so you can see which role is inflating the window — subagent nodes show that subagent's **own** window, badged with its id
  - every node has the same three collapsible sections — **Outputs**, **Inputs**, **Attributes**
  - in **Outputs**, **Tool Dispatch** nodes render a status badge (**✓ success** / **✗ error**), the command that ran, and the output/error; **LLM Call** nodes render the response text, reasoning, and tool calls — each in a mini-editor
  - in **Inputs**, **LLM Call** nodes show both the `messages` sent to the model and the `tools` definitions the model was given that turn (the tool schemas that reached the LLM, not just the calls it suggested)
  - every content box (JSON and raw text) is a **mini VS Code-style editor** (powered by CodeMirror, vendored offline): syntax highlighting, line numbers, and code folding; a clickable **schema breadcrumb** for JSON (e.g. `root › [0] › function`); **collapse all / expand all**, **copy all**, and **find** with keyboard next/previous (Enter / Shift+Enter)
- The current **session id** is shown in the header (click to copy); the picker dropdown lists all sessions
- Loop-detected tool calls are flagged inline
- Detected failure streaks are flagged inline too (a distinct **anomaly** tag on the affected tool calls, alongside a dedicated anomaly node reporting the streak's length, signature, and tokens spent) — separate from the loop flag
- A **Breakdown** toggle in the stats bar (shown whenever the session has token-usage data) reveals per-call-kind and per-agent token totals across the whole delegated tree
- When a run used [skills](#skills), `use_skill` tool calls carry a **🧠 skill** badge and the stats bar shows an **offered / loaded** count; traces recorded before skills existed load and render unchanged

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Running local LLM server (MLX Server, Ollama, etc.)

## Architecture & Contributing

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the module layout, agent loop, tool dispatch, context handoff, and session persistence.
- [`CONTRIBUTE.md`](CONTRIBUTE.md) — development standards, tooling, testing, and commit conventions.

## Documentation

Build the API reference locally:

```bash
uv run sphinx-build -W -b html docs docs/_build/html
```

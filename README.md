# my-coding-agent

A personal, from-scratch Python agent framework for running LLM-powered coding agents against local LLM servers. Built for me.

## What It Is

A self-contained agentic loop — no external agent frameworks. The agent calls a local model, executes tool calls, feeds results back, and repeats until the task is done or a step/context limit is hit.

Key ideas:
- **Local-first**: targets OpenAI-compatible endpoints (MLX Server, Ollama) running on your machine
- **Node-based pipeline**: the agentic loop is a DAG of named nodes (`ContextGuard → ToolRouting → LLMCall → ToolDispatch → AnomalyDetect → FinalizeStep`) with an explicit data contract (`RunContext`) flowing between them
- **Runtime anomaly detection**: while the run is live, a same-class tool-failure streak (e.g. the same tool failing 3+ times in a row with the same error class, regardless of args) is flagged the moment it happens — logged as a warning and recorded in the session's event stream, in main agents and subagents alike
- **Dangerous-command refusal gate**: every `bash` call is checked against a deterministic, local rule set before it runs (recursive root/home deletes, remote-content-piped-to-shell, raw-device writes, fork bombs, permission blasts, credential exfiltration, destructive git force-pushes); a match never reaches the shell — the model gets back a structured refusal (reason + security-standard reference + safer alternative) so it can steer, and the refusal is logged and recorded for later review. This is a high-signal first layer, not a complete boundary — obfuscated commands can evade textual matching by design, which is why the harness also records each run's protection posture (`screened_only` vs. `sandboxed`, see [`SECURITY.md`](SECURITY.md)) rather than implying completeness. Extensible and can be disabled with `--no-safety-gate` if you really need to
- **Exfiltration guard**: before an outbound tool call (currently `fetch_web`) sends its payload, it's checked deterministically and locally against well-known secret paths (`.env`, SSH keys, cloud credentials, `.netrc`, `*.pem`/`*.key`) and content signatures (PEM private-key headers, common token formats); a match never reaches the network — the model gets back a structured block naming only the matched category, never the secret value, and the block is logged and recorded for later review. Disabled runs are byte-identical to today; can be disabled with `--no-exfil-guard` if you really need to
- **Network egress filter**: every `fetch_web` destination is checked against an actively-maintained, open-source blocklist of publicly-catalogued malicious domains (hagezi Threat-Intelligence-Feeds, cached and refreshed offline-tolerantly) before the connection proceeds; a known-bad host is denied with a structured, steerable block instead of connecting, an unknown host is unaffected. See [`SECURITY.md`](SECURITY.md); disable with `--no-egress-filter` if you really need to
- **OS-level bash sandbox** (opt-in, macOS): pass `--sandbox` and every `bash` subprocess runs inside an Apple Seatbelt sandbox (`sandbox-exec` with a generated profile) — filesystem writes are default-deny outside the workspace directory plus a small temp allowlist, and all outbound network is default-deny, enforced by the OS beneath the shell regardless of how the command is phrased. Off by default (byte-identical to a non-sandboxed run); enabling it on a host without `sandbox-exec` (i.e. non-macOS) refuses to run `bash` rather than silently executing unconfined. This is what the refusal gate's protection posture reports as `sandboxed`
- **Untrusted content confinement**: content the agent pulls from outside the task itself — currently `fetch_web` results — is tagged untrusted at the point it enters the run and wrapped in explicit "this is data, not instructions" framing where it's inserted into the conversation (never the system prompt, so prompt caching is unaffected); once untrusted content is in play, a build/install/setup script (`npm install`, `make`, `sh install.sh`, `./configure`, ...) run against a repo the agent itself cloned this run is refused rather than auto-run at full privilege — a defense-in-depth pair of measures against indirect prompt injection, not a complete guarantee. A run that never fetches or clones anything is unaffected
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
my-coding-agent = "my_coding_agent.cli:main"
```

It points to `main()` in `src/my_coding_agent/cli.py`. To change the default command, edit that mapping (and run `uv sync` to refresh the script); to change the default behaviour, edit the Click options on `main()`.

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
| `--no-safety-gate` | `False` | Disable the dangerous-command refusal gate for this run |
| `--no-egress-filter` | `False` | Disable the network egress filter for this run (same effect as `MCA_DISABLE_EGRESS_FILTER=1`); see [`SECURITY.md`](SECURITY.md) |
| `--no-exfil-guard` | `False` | Disable the secret-exfiltration guard for this run (same effect as `MCA_DISABLE_EXFIL_GUARD=1`); see [`SECURITY.md`](SECURITY.md) |
| `--sandbox` | `False` | Run every `bash` subprocess inside an OS-level sandbox (macOS Seatbelt); see [Sandboxing](#sandboxing) |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMLX_API_URL` | `http://127.0.0.1:8321/v1` | Local LLM API base URL |
| `OMLX_API_KEY` | `changeme` | API key (usually ignored by local servers) |
| `OMLX_MODEL` | `Qwen3.6-35B-A3B-6bit` | Model ID to use |
| `MCA_TOOL_MAX_CONCURRENCY` | `4` | Max read-only tool calls overlapped per assistant message (`1` disables overlap) |
| `MCA_LLM_OUTAGE_TOLERANCE_S` | `300` | Seconds the client keeps probing a stalled/restarting LLM server (transport / HTTP 5xx / 429) before giving up with a classified, resumable stop; other 4xx fail fast |
| `MCA_SUPERSESSION` | `1` | Set to `0` to disable retiring provably-superseded tool results (restores append-only conversation behavior byte-for-byte) |
| `MCA_BASH_SANDBOX` | unset (off) | Set to `1` for the same effect as `--sandbox`; a shell-exported value and the flag are equivalent |
| `MCA_HOOKS_CONFIG` | unset | Path to a JSON file of lifecycle hooks (see [Hooks](#hooks)); unset means no hooks — byte-identical to a run without this feature |

## Sandboxing

Pass `--sandbox` and every `bash` subprocess for that run — including a delegated subagent's, which shares the same process — runs inside an Apple Seatbelt sandbox (`sandbox-exec` with a profile generated fresh per run):

- **Filesystem**: writes are default-deny outside the workspace directory (the `bash` tool's `base_dir`, normally the CWD) plus a small temp allowlist (`/tmp`, `/private/tmp`, `$TMPDIR`); reads are allowed broadly (system dirs and dependencies tools need to run). The scope is fixed at the start of the run and never expands based on what a command asks for. The allowlist deliberately excludes macOS's broader per-user temp *parent* (e.g. `/private/var/folders`) — that directory is shared machine-wide across every process's own temp subdirectory, so allowing the whole parent would let a sandboxed command write into another process's temp space, defeating the point of the scope.
- **Network**: all outbound network access from the subprocess is denied by default. (A future egress allow/deny policy can layer selective re-permission on top of this boundary — not yet implemented.)
- **Mechanism**: macOS only, via `sandbox-exec` — the same approach Claude Code's sandbox-runtime and OpenAI Codex ship. Off by default; a run without the flag is byte-identical to today, with no sandbox wrapping and no sandbox events.
- **Degradation**: enabling `--sandbox` on a host without `sandbox-exec` (non-macOS, or the binary missing) refuses to run `bash` — every `bash` call fails with an explicit error — rather than silently executing unconfined.
- **Observability**: sandbox activation (once per run) and each denied command are recorded as passive rows in `events.jsonl` (`sandbox_activation` / `sandbox_denial`), same template as the refusal gate's events — never influences enforcement, which the OS already applied unconditionally.

A denied command surfaces to the model as an ordinary failed `bash` call (`ok: false`, non-zero `exit_code`) — no approval prompt, no notification loop; the model reads the failure and steers.

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


## Hooks

Lifecycle hooks are a deterministic seam where developer-configured, in-process Python code can observe — or, for `PreToolUse`, veto — what happens at defined points in a session, independent of what the model decides (issue #129).

**Events** (fired by the engine, never by the model): `PreToolUse` (before a tool call is dispatched), `PostToolUse` (after it completes), `SessionStart` (once, at the start of a run), `SessionEnd` (once, at the end).

**Veto contract** — a `PreToolUse` hook may return a block decision that prevents the tool from running at all (no subprocess, no side effect). The call resolves to the standard `ok:false` envelope — `error` carries model-facing prose and `metadata.hook_block` carries the hook's name and reason — reusing the same no-execution path the dangerous-command refusal gate (`SECURITY.md`) already uses, as a `blocked_by_hook` sibling of `refused`/`not_found`/`wrong_args`/`raised`. Any other event's hooks are observe-only.

**Configuration** — set `MCA_HOOKS_CONFIG` to a JSON file listing hook entries:

```json
[
  {
    "name": "block-dangerous-sentinel",
    "event": "PreToolUse",
    "tool": "bash",
    "callable": "my_project.hooks:deny_sentinel"
  }
]
```

`callable` is a `"module:attribute"` reference to a Python function taking a `HookContext` and returning `HookResult.block(reason)` to veto, or `None`/`HookResult.allow()` to let the call proceed. `tool` (optional) restricts a tool-event hook to one tool name. Hooks fire in the order listed; a `PreToolUse` block short-circuits any hooks after it. A malformed entry is skipped with a warning, never fails the run; a hook that raises is caught, logged, and treated as `allow` (fail-open — a buggy observability hook must not stall a run). No `MCA_HOOKS_CONFIG` set means zero hooks load, so behavior, prompts, and tool schemas are byte-identical to a run without this feature.

Every firing (allowed or blocked) is recorded as a passive `hook` row in `events.jsonl`; hook-free runs emit none.

## Observability

Every run automatically records a structured `events.jsonl` alongside the other session files under `.my_coding_agent/<session_id>/`. No configuration needed — capture is always on.

| File | Contents |
|---|---|
| `stderr.log` | Plain-text log of the full run |
| `session_data.json` | Metrics, tool records, LLM call log, stop reason, usage rollup (own + delegated subagents, per call kind) |
| `events.jsonl` | Structured event stream (LLM calls, tool I/O, handoffs, skill-index offers, outage waits/recovery, dangerous-command refusals, exfiltration-guard blocks, the run's protection posture, lifecycle hook firings) |
| `checkpoint.json` | Engine-owned per-step resume checkpoint (exact `messages` + step/token counters), written atomically after each completed step; consumed by `--resume` |
| `artifacts/<tool_call_id>.<stream>.txt` | Full content of each offloaded large output stream (`stdout`/`stderr`), written at creation so bash can skim it during the run |
| `tool_artifacts.json` | End-of-run audit dump of the in-memory artifact records |

### Machine-wide resource capture

Since the LLM runs locally, machine load *is* the run's real cost. A background sampler (no sudo required) records **machine-wide** RAM/CPU/GPU/network/disk figures — what the Mac was doing, not per-process attribution — for the execution window of every timed event (`llm_call`, `tool_call`, `summarizer`), and a session-wide rollup (peaks/averages, byte totals) is persisted in `session_data.json` alongside the token/time totals. These figures are explicitly labeled `machine_wide: true` since other processes on the machine (e.g. a browser) count toward them too. Capture is passive (never throttles or aborts the run) and degrades gracefully: if the GPU reading is unavailable (non-macOS, or the `ioreg` output changes) or the sampler ever fails, the affected field (or all `resources` data) is simply absent — the run and its trace are unaffected. Sessions recorded before this existed load and render unchanged. Both the terminal run summary and the Trace Explorer (node detail, session stats) show these figures whenever they were captured.

### Web UI

One local, offline server hosts the interface behind a persistent navigation bar, with **Traces** as its only destination:

```bash
my-coding-agent-webui          # defaults: port 7474, dir .my_coding_agent
my-coding-agent-webui --port 8080 --dir /path/to/.my_coding_agent
my-coding-agent-webui --dir .my_coding_agent --check <session_id>  # deterministic sum-check, no server; exit 0/1
```

Then open `http://localhost:7474`. It remembers where you left off — last-visited tab and in-progress selection (e.g. the open session) persist to a local SQLite store under `.my_coding_agent/webui/` and are restored the next time you launch it, even after stopping the process.

This supersedes the old standalone `my-coding-agent-traces` entrypoint (removed); its Trace Explorer rendering is unchanged, just mounted into the **Traces** tab below.

#### Trace Explorer

The **Traces** tab visualises sessions with an interactive pipeline DAG. The UI (an Apple-minimalist Preact app, served fully offline) shows:

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
- A tool call refused by the dangerous-command gate carries its own distinct **refused** tag (tree row + detail header, alongside — never replacing — loop/anomaly) and a **🛑 refused** count in the stats bar; its detail panel spells out the reason, the matched rule, clickable security-standard reference links, and the safer alternative
- The stats bar shows the run's **protection posture** — a distinct **🔒 sandboxed** or **🛡 screened only** tag — so you can tell at a glance whether OS-level enforcement was in force or only the textual gate; traces recorded before posture capture existed show no tag and render unchanged
- A **Breakdown** toggle in the stats bar (shown whenever the session has token-usage data) reveals per-call-kind and per-agent token totals across the whole delegated tree
- When a session was produced by an eval run (a `verdict.json` sibling exists next to its trace), the stats bar shows an at-a-glance **✓ pass** / **✗ fail** tag and a **Verdict** toggle revealing the run id, case id, metrics, and detail/rationale; sessions with no verdict render with no verdict UI at all
- When a run used [skills](#skills), `use_skill` tool calls carry a **🧠 skill** badge and the stats bar shows an **offered / loaded** count; traces recorded before skills existed load and render unchanged

## Evals

Run the agent against a fixed, named set of tasks and get back a repeatable, scored result — instead of "I tried it and it looked good".

An eval **case** is a plain JSON file: a task prompt, a scorer ref, and the scorer's expected/threshold data.

```json
{
  "id": "hello_world",
  "task": "Say exactly the word 'pong' and nothing else.",
  "scorer": "exact_match",
  "expected": { "contains": "pong" }
}
```

Point the runner at a directory of your own case files (`--cases <dir>`; the default is `.my_coding_agent/evals/cases/`). No example cases ship committed, so the bare invocation exits `1` with "No eval cases found" until you add case files to the default directory:

```bash
my-coding-agent-eval --cases path/to/case/dir
my-coding-agent-eval               # reads .my_coding_agent/evals/cases/; exits 1 while it's empty
```

Each case runs the agent in a fresh, isolated temp workspace (so cases can't contaminate each other or the real repo), collects its trace, and scores it with the case's scorer (`evals.scoring.register_scorer` is the extension point for scorers). Three ship today: `exact_match`, the baseline deterministic scorer (`equals`/`contains` check on the final output); `trajectory`, which scores the run's *path* rather than its answer — tool-selection correctness, argument validity, error handling, and efficiency (steps/tokens/wall-clock, redundancy), each reported as its own dimension so a weak run is locatable, not just a single number; and `judge`, the rubric-based LLM judge — it grades the output against a declared disk-loaded rubric (criteria, score scale, per-criterion anchors), never a free-form "is this good?" prompt. A versioned, self-describing result record — run identity (agent/model version, dataset ref, timestamp), per-case scores, and aggregate metrics — is written under `.my_coding_agent/evals/<run_id>/result.json`. The full verdict — per-check pass/fail, score, and rationale/detail — prints to the terminal at run end (not just the summary pass-rate line), and each score also carries the session id of the agent run that produced it, so it's traceable back to the run's evidence rather than only its aggregate result.

Every score names the session id of the agent run that produced it (`EvalScore.session_id`, additive — older records without it load with `None`). For a run whose session directory survives past the run (the real-cwd paths: `run_evaluation` and config-driven runs; the case runner's temp-workspace sessions are torn down before the result is written and are exempt by design), a `verdict.json` — `{run_id, case_id, passed, metrics, detail, result_path}` — is written into `.my_coding_agent/<session_id>/` alongside the trace, so a verdict is reachable starting from only a session id without scanning every result record. It's a plain sibling file; `events.jsonl`/`session_data.json` are never touched, and the write is fault-tolerant (a failure logs a warning and never fails the run).

### Datasets

A **dataset** groups cases into a named, versioned collection, so a result records exactly which cases (and which version of that set) it ran against — a static "golden set" stops meaning anything once it's memorised, and a dataset's version lets a later comparison tell whether two runs are even comparable. Datasets live under `.my_coding_agent/evals/datasets/<dataset_id>/`. Nothing ships committed there, so you create your own from case ids that reference `*.json` files under your cases directory:

```python
from my_coding_agent.evals.datasets import create_dataset, add_case, retire_case, run_dataset, load_dataset

create_dataset("smoke", ["hello_world"])    # ids of *.json case files under .my_coding_agent/evals/cases/
add_case("smoke", "another_case")       # bumps to version 2
retire_case("smoke", "hello_world")     # bumps to version 3; version 1/2 still loadable

dataset = load_dataset("smoke")         # latest version by default
result = run_dataset(dataset)           # result.dataset == "smoke@v3"
```

### Comparing runs and gating CI

`my-coding-agent-eval compare` turns two persisted result records into a pass/fail verdict a CI pipeline can enforce — advisory to a human merge decision, never an auto-merge:

```bash
my-coding-agent-eval compare <baseline_run_id> <candidate_run_id>
my-coding-agent-eval compare <baseline_run_id> <candidate_run_id> --floor pass_rate=0.9
```

Each argument is either a run id under `.my_coding_agent/evals/` or a path to a result directory. The comparison reports per-metric deltas *and* which individual cases flipped pass↔fail, so a flat aggregate can't hide a subset regression. It refuses to compare two runs stamped with different dataset id/version (`--allow-cross-version` downgrades this to a loud warning instead). A configurable `--floor METRIC=VALUE` (repeatable) and the default "no previously-passing case regressed" rule turn the comparison into a verdict; the command exits `0` on pass and non-zero on a regression — the same exit-code pattern as `my-coding-agent-webui --check` — always naming the violated floor or regressed case rather than failing silently.

`add_failure_case` turns a recorded run failure into a new regression case file and adds it to a dataset in one step. Every mutation appends a new version rather than rewriting history, so `load_dataset("smoke", version=1)` still recovers the original membership.

### Declarative YAML run config

A single YAML file can fully define an eval/pipeline run — connection, prompts, run parameters, and evaluation criteria — as one versionable, plain-text artifact, so it can be read, diffed, and edited without a running server or the web UI:

```yaml
llm:
  api_url: http://127.0.0.1:8321/v1   # optional; falls back to OMLX_API_URL, then the documented default
  model: Qwen3.6-35B-A3B-6bit         # optional; falls back to OMLX_MODEL, then the documented default
  api_key_env: MY_API_KEY             # optional; names an env var — never a raw key — falls back to OMLX_API_KEY
  timeout: 30                        # optional; seconds, falls back to the documented default

run:
  system_prompt: "You are a coding assistant."   # optional
  task: "Say exactly the word 'pong' and nothing else."
  max_steps: 20                                  # optional; defaults to the agent's own default

evaluation:
  checks:                              # inline checks, scored against this run's own output
    - name: says-pong
      evaluator: exact_match
      expected: { contains: "pong" }
```

`llm.api_key` (a raw secret value) is rejected at validation time — the file must stay safe to commit, so only `api_key_env` (an environment variable *name*) is accepted. `evaluation.checks` must declare at least one check.

A runnable copy lives at [`examples/eval_run_config.yaml`](examples/eval_run_config.yaml):

```bash
my-coding-agent-eval run --config examples/eval_run_config.yaml
my-coding-agent-eval run --config path/to/your/run.yaml
```

The run executes in the real working directory (unlike the isolated-workspace case runner): its session lands under `.my_coding_agent/`, visible in the **Traces** tab. A validation failure (malformed YAML, unknown key, missing field, unknown evaluator, or a raw API key) prints every problem found and exits `2` without starting an agent run; a scored run exits `0` on pass, `1` on fail. The written result record (`.my_coding_agent/evals/<run_id>/result.json`) carries the config file's path and a content hash, so it stays traceable to the exact configuration version that produced it.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Running local LLM server (MLX Server, Ollama, etc.)

## Architecture & Contributing

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the module layout, agent loop, tool dispatch, context handoff, and session persistence.
- [`CONTRIBUTE.md`](CONTRIBUTE.md) — development standards, tooling, testing, and commit conventions.
- [`SECURITY.md`](SECURITY.md) — the dangerous-command refusal gate (how to disable it, `--no-safety-gate` / `MCA_DISABLE_DANGEROUS_COMMAND_GATE`, and how to extend its rule set) and the network egress filter (how to disable it, `--no-egress-filter` / `MCA_DISABLE_EGRESS_FILTER`, and how to pick a blocklist source).
- [Sandboxing](#sandboxing) — the opt-in OS-level `bash` sandbox (`--sandbox` / `MCA_BASH_SANDBOX`): scope, mechanism, and degradation.

## Documentation

Build the API reference locally:

```bash
uv run sphinx-build -W -b html docs docs/_build/html
```

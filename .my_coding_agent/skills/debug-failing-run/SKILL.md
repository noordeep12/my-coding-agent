---
name: debug-failing-run
description: Diagnose a failed or surprising agent run using its session files (events.jsonl, session_data.json) and the Trace Explorer.
---

Every run records session files under `.my_coding_agent/<session_id>/`. Use them
to diagnose a failure instead of guessing from the terminal output alone.

1. **Find the session.** List `.my_coding_agent/` and pick the newest
   `<session_id>` directory (or the one the failing run printed).

2. **Read the structured event stream.** `events.jsonl` is one JSON object per
   line — the LLM calls, tool I/O, handoffs, and skill-index offers in order.
   Scan for the failing `tool_call` (non-zero exit, error envelope) or the
   `llm_call` where the run went off the rails. Full offloaded tool output lives
   under `artifacts/<tool_call_id>.<stream>.txt`.

3. **Check the rollup.** `session_data.json` holds metrics, the tool records, the
   LLM call log, the stop reason, and the usage/resource rollup — start here for
   *why the run stopped* and where time/tokens went.

4. **Open the Trace Explorer** for a visual, clickable view:
   ```
   my-coding-agent-traces                 # port 7474, dir .my_coding_agent
   my-coding-agent-traces --port 8080 --dir /path/to/.my_coding_agent
   ```
   Click into the suspect node to see its inputs, outputs, timing, and resource
   figures. `use_skill` calls carry a 🧠 badge and the stats bar shows the
   offered/loaded skill count.

5. **Deterministic check, no server:**
   `my-coding-agent-traces --dir .my_coding_agent --check <session_id>` runs a
   sum-check and exits 0/1 — handy in CI or a quick sanity pass.

Traces recorded before a given feature existed still load and render unchanged,
so old sessions are safe to inspect.

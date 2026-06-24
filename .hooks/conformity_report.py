#!/usr/bin/env python3
"""Stop hook: produce a conformity report for the code changed in this implementation.

Triggered when Claude Code finishes a turn. If code under src/ changed versus HEAD, a
headless `claude -p` auditor applies the gap-audit criteria scoped to the changed files,
then conformity.md is written with the verdict and a diff-bound meta block.

Fail-closed: if the auditor cannot run, the report state is 'blocked' so the pre-commit
gate keeps the user's final word intact.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformity import CODE_PATH, code_diff, diff_hash, report_path

# The auditor below is itself a `claude` process whose own Stop hook would re-enter
# here — this env flag breaks that recursion.
if os.environ.get("CONFORMITY_HOOK_RUNNING"):
    sys.exit(0)

try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    payload = {}
if payload.get("stop_hook_active"):
    sys.exit(0)

current_hash = diff_hash()
if not current_hash:
    sys.exit(0)  # no code changed

changed = subprocess.run(
    ["git", "diff", "--name-only", "HEAD", "--", CODE_PATH],
    capture_output=True,
    text=True,
).stdout.strip()

INSTRUCTIONS = (
    "You are a conformity auditor for this repository. Audit ONLY the code changes "
    "in the diff below against the project policy documents: .claude/CLAUDE.md, "
    "CONTRIBUTE.md, ARCHITECTURE.md, README.md. Read those documents as needed.\n\n"
    "Report ONLY deviations (gaps) present in the changed code. Pure observation — "
    "no fixes, no suggestions, no praise. Present tense, indicative mood.\n\n"
    "Severity: CRITICAL (foundational rule / security / data loss / silent wrong "
    "behavior), HIGH (explicit stated standard violated), MEDIUM (documented "
    "style/best-practice deviation), LOW (minor inconsistency).\n\n"
    "For each gap use this markdown block:\n\n"
    "### G-NN — <title>\n\n"
    "| | |\n|---|---|\n"
    "| **Severity** | CRITICAL / HIGH / MEDIUM / LOW |\n"
    "| **Criticality** | <one sentence on urgency> |\n\n"
    "**What was identified:** <factual observation>\n"
    "**Why this gap was identified:** <which policy rule triggered it>\n"
    "**Why it matters:** <concrete consequence>\n"
    "**Problem location:** `path:line` — <what is there>\n"
    '**Standard location:** `DOC §sec` — "<quoted rule>"\n\n'
    "Output rules — follow EXACTLY:\n"
    "- The VERY FIRST line of your output must be either `CONFORMITY_VERDICT: PASS` "
    "(no gaps) or `CONFORMITY_VERDICT: GAPS` (one or more gaps).\n"
    "- After that line, output the gap blocks (or nothing if PASS)."
)

PROMPT = f"{INSTRUCTIONS}\n\nChanged files:\n{changed}\n\nDiff:\n{code_diff()}\n"


def run_auditor():
    env = {**os.environ, "CONFORMITY_HOOK_RUNNING": "1"}
    try:
        proc = subprocess.run(
            ["claude", "-p", PROMPT, "--allowedTools", "Read Grep Glob"],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
    except FileNotFoundError:
        return None, "claude CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return None, "auditor timed out"
    if proc.returncode != 0:
        return None, f"auditor exited {proc.returncode}: {proc.stderr.strip()[:200]}"
    return proc.stdout.strip(), None


output, error = run_auditor()
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

if error is not None:
    state = "blocked"
    body = (
        f"> Auditor unavailable: {error}\n>\n"
        "> Review the changes manually, then set state to 'approved' to commit."
    )
else:
    first, _, rest = output.partition("\n")
    if first.strip() == "CONFORMITY_VERDICT: PASS":
        state = "pass"
        body = rest.strip() or "No conformity gaps found in the changed code."
    else:
        state = "blocked"
        body = rest.strip() if first.startswith("CONFORMITY_VERDICT:") else output

changed_lines = os.linesep.join(f"> - `{f}`" for f in changed.splitlines())
report = f"""# Conformity Report
<!-- conformity-meta
diff_hash: {current_hash}
state: {state}
-->

> **Generated:** {now}
> **Scope:** code changed versus HEAD under `{CODE_PATH}`
> **Changed files:**
{changed_lines}

---

{body}
"""

with open(report_path(), "w", encoding="utf-8") as fh:
    fh.write(report)

print(f"Conformity report written to conformity.md (state: {state}).", file=sys.stderr)
sys.exit(0)

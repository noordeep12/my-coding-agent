#!/usr/bin/env python3
"""Conformity hooks — one entry point, two modes.

    python3 .hooks/check_conformity_report.py            # pre-commit gate (default)
    python3 .hooks/check_conformity_report.py --report   # Claude Code Stop hook

Gate mode blocks a commit whose code changes lack an approved conformity report.
Report mode audits the changed code (headless ``claude -p``, gap-audit criteria) and
writes conformity.md. The user always has the final word: a 'blocked' report stays
blocked until the user reviews it and sets state to 'approved'.

conformity.md carries a machine-readable meta block read by the gate:

    <!-- conformity-meta
    diff_hash: <sha256 of `git diff HEAD -- src/`>
    state: pass | blocked | approved
    -->
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPORT_FILE = "conformity.md"
CODE_PATH = "src/"

_META_RE = re.compile(r"<!--\s*conformity-meta\s*(.*?)-->", re.DOTALL)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() or os.getcwd()


def code_diff():
    """Diff of tracked code under src/ versus HEAD (staged + unstaged)."""
    out = subprocess.run(
        ["git", "diff", "HEAD", "--", CODE_PATH],
        capture_output=True,
        text=True,
    )
    return out.stdout


def diff_hash():
    """sha256 of the code diff; empty string when there is no code change."""
    diff = code_diff()
    if not diff.strip():
        return ""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def report_path():
    return os.path.join(repo_root(), REPORT_FILE)


def parse_report():
    """Return {'diff_hash':..., 'state':...} or None if absent/unparseable."""
    path = report_path()
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = _META_RE.search(text)
    if not m:
        return None
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    if "diff_hash" not in meta or "state" not in meta:
        return None
    return meta


# --------------------------------------------------------------------------- #
# Gate mode (pre-commit)
# --------------------------------------------------------------------------- #
def run_gate():
    current = diff_hash()
    if not current:
        return 0  # no code changes under src/ — nothing to gate

    report = parse_report()
    if report is None:
        print(f"No conformity report ({REPORT_FILE}) for the changed code.")
        return 1

    if report["diff_hash"] != current:
        print(f"{REPORT_FILE} is stale: code changed since the last report.")
        print("Regenerate it (the Stop hook does this at end of implementation).")
        return 1

    if report["state"] not in ("pass", "approved"):
        print(f"Conformity report state is '{report['state']}' — commit blocked.")
        print(f"Review {REPORT_FILE}; set state to 'approved' to sign off.")
        return 1

    return 0


# --------------------------------------------------------------------------- #
# Report mode (Claude Code Stop hook)
# --------------------------------------------------------------------------- #
def _build_prompt(changed):
    instructions = (
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
        "- The VERY FIRST line of your output must be either "
        "`CONFORMITY_VERDICT: PASS` (no gaps) or `CONFORMITY_VERDICT: GAPS`.\n"
        "- After that line, output the gap blocks (or nothing if PASS)."
    )
    return f"{instructions}\n\nChanged files:\n{changed}\n\nDiff:\n{code_diff()}\n"


def _run_auditor(prompt):
    env = {**os.environ, "CONFORMITY_HOOK_RUNNING": "1"}
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--allowedTools", "Read Grep Glob"],
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


def run_report():
    # The auditor below is itself a `claude` process whose own Stop hook would
    # re-enter here — this env flag breaks that recursion.
    if os.environ.get("CONFORMITY_HOOK_RUNNING"):
        return

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if payload.get("stop_hook_active"):
        return

    current_hash = diff_hash()
    if not current_hash:
        return  # no code changed

    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", CODE_PATH],
        capture_output=True,
        text=True,
    ).stdout.strip()

    output, error = _run_auditor(_build_prompt(changed))
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
    print(f"Conformity report: {REPORT_FILE} (state: {state}).", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if "--report" in sys.argv[1:]:
        run_report()
        return 0
    return run_gate()


if __name__ == "__main__":
    sys.exit(main())

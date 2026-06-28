#!/usr/bin/env python3
"""Conformity hook — interactive gap review via AskUserQuestion.

    python3 .hooks/check_conformity_report.py              # pre-commit gate
    python3 .hooks/check_conformity_report.py --report     # Claude Code Stop hook
    python3 .hooks/check_conformity_report.py --set-mode MODE       # auto|interactive
    python3 .hooks/check_conformity_report.py --disposition PAIRS   # G-01=exception

Stop hook flow
--------------
1. Run headless audit of changed code against policy docs.
2. PASS → mark state resolved, exit 0 (Claude stops normally).
3. GAPS found → exit 2 (blocks Claude from stopping) and write to stderr an
   instruction that tells the active Claude session to call AskUserQuestion
   for each gap. Claude MUST NOT auto-fix; it MUST ask the user first.
4. After user decisions, Claude calls --set-mode and --disposition to record
   choices, then the next Stop hook run re-audits.

Auto mode
---------
If the user selects "Auto-fix", Claude is permitted to fix gaps autonomously
for up to MAX_AUTO_ITERATIONS rounds before switching to interactive review.

State is persisted in .hooks/.conformity_state.json (gitignored).
"""

import hashlib
import json
import os
import re
import subprocess
import sys

CODE_PATH = "src/"
STATE_FILE = ".hooks/.conformity_state.json"
MAX_AUTO_ITERATIONS = 3

_GAP_RE = re.compile(r"^###\s+(G-\d+)\s+[—–-]+\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() or os.getcwd()


def code_diff():
    out = subprocess.run(
        ["git", "diff", "HEAD", "--", CODE_PATH],
        capture_output=True,
        text=True,
    )
    return out.stdout


def diff_hash():
    diff = code_diff()
    if not diff.strip():
        return ""
    return hashlib.sha256(diff.encode()).hexdigest()


def state_path():
    return os.path.join(repo_root(), STATE_FILE)


def read_state():
    p = state_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_state(state):
    p = state_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


MAX_STOP_HOOK_RUNS = 2


def fresh_state(hash_):
    return {
        "diff_hash": hash_,
        "status": "pending",
        "mode": None,
        "auto_iterations": 0,
        "stop_hook_runs": 0,
        "dispositioned_gaps": [],
    }


def parse_gaps(text):
    """Return list of (gap_id, title) tuples from auditor output."""
    return [(m.group(1), m.group(2).strip()) for m in _GAP_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


def _build_audit_prompt(changed):
    changed_files = "\n".join(f"  {f}" for f in changed.splitlines())
    return (
        "You are a conformity auditor for this repository.\n\n"
        "SCOPE — audit ONLY:\n"
        "1. The diff provided below.\n"
        f"2. The changed files listed here (you may Read these if you need "
        f"more context beyond the diff):\n{changed_files}\n\n"
        "DO NOT read any other source files. Do not follow imports, callers, "
        "or related modules outside the changed files.\n\n"
        "Policy documents you MUST read to audit against: "
        ".claude/CLAUDE.md, CONTRIBUTE.md, ARCHITECTURE.md, README.md.\n\n"
        "Report ONLY deviations (gaps) present in the changed code. Pure "
        "observation — no fixes, no suggestions, no praise.\n\n"
        "Severity: CRITICAL (foundational rule / security / data loss / silent "
        "wrong behavior), HIGH (explicit stated standard violated), MEDIUM "
        "(documented style/best-practice deviation), LOW (minor inconsistency).\n\n"
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
        "- After that line, output the gap blocks (or nothing if PASS).\n\n"
        f"Changed files:\n{changed}\n\nDiff:\n{code_diff()}"
    )


def _run_auditor(changed):
    env = {**os.environ, "CONFORMITY_HOOK_RUNNING": "1"}
    try:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                _build_audit_prompt(changed),
                "--allowedTools",
                "Read Grep Glob",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
    except FileNotFoundError:
        return None, "claude CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return None, "auditor timed out after 300 s"
    if proc.returncode != 0:
        return None, f"auditor exited {proc.returncode}: {proc.stderr.strip()[:200]}"
    return proc.stdout.strip(), None


# ---------------------------------------------------------------------------
# Stop hook (--report)
# ---------------------------------------------------------------------------


def _block(message):
    """Block Claude from stopping; feed message back to the active session."""
    print(message, file=sys.stderr)
    sys.exit(2)


def _interactive_instruction(
    remaining_gaps, full_body, mode_unknown, auto_exhausted=False
):
    gap_questions = []
    for gid, title in remaining_gaps:
        gap_questions.append(f"""    {{
      "question": "{gid}: {title}",
      "header": "{gid}",
      "multiSelect": false,
      "options": [
        {{"label": "Fix", "description": "Implement the conformity fix now"}},
        {{"label": "Exception", "description": "Accept as intentional deviation"}},
        {{"label": "Skip", "description": "Defer — do not fix in this session"}}
      ]
    }}""")

    mode_question = ""
    if mode_unknown:
        mode_question = """    {
      "question": "How would you like to handle these conformity gaps?",
      "header": "Mode",
      "multiSelect": false,
      "options": [
        {"label": "Interactive", "description": "Review each gap and decide"},
        {"label": "Auto-fix", "description": "Auto up to 3 rounds then interactive"}
      ]
    },
"""

    auto_note = ""
    if auto_exhausted:
        auto_note = (
            f"\nAuto-fix exhausted ({MAX_AUTO_ITERATIONS} rounds). "
            "Switching to interactive review.\n"
        )

    all_questions = mode_question + ",\n".join(gap_questions)
    n = len(remaining_gaps)

    disposition_examples = ",".join(f"{gid}=exception" for gid, _ in remaining_gaps[:2])

    return f"""CONFORMITY HOOK — INTERACTIVE REVIEW REQUIRED{auto_note}

{n} gap(s) need your attention. YOU MUST CALL AskUserQuestion NOW.
DO NOT auto-fix. DO NOT write a text response. CALL AskUserQuestion IMMEDIATELY.

━━━ AskUserQuestion parameters — call this tool right now ━━━

{{
  "questions": [
{all_questions}
  ]
}}

━━━ After the user answers ━━━

If mode question was shown:
  - "Interactive" → python3 .hooks/check_conformity_report.py --set-mode interactive
  - "Auto-fix"    → python3 .hooks/check_conformity_report.py --set-mode auto

For each gap answer:
  - "Fix"       → implement the fix (Stop hook re-audits on next run)
  - "Exception" → python3 .hooks/check_conformity_report.py --disposition G-NN=exception
  - "Skip"      → python3 .hooks/check_conformity_report.py --disposition G-NN=skip
  (replace G-NN with the actual gap ID, e.g. {disposition_examples})

Once all gaps are fixed, excepted, or skipped the next Stop hook run will
detect PASS and let Claude stop normally.

━━━ Full gap details ━━━

{full_body}
"""


def _dispatch_gaps(state, remaining, body):
    """Block Claude with the appropriate instruction for the current gap state."""
    if state["mode"] == "auto" and state["auto_iterations"] < MAX_AUTO_ITERATIONS:
        state["auto_iterations"] += 1
        write_state(state)
        gaps_list = "\n".join(f"  - {gid}: {title}" for gid, title in remaining)
        n = state["auto_iterations"]
        _block(
            f"CONFORMITY HOOK — AUTO-FIX MODE (round {n}/{MAX_AUTO_ITERATIONS})\n\n"
            f"Fix the following gaps now:\n{gaps_list}\n\n"
            f"After fixing, do NOT call AskUserQuestion — just finish. "
            f"The Stop hook will re-audit.\n\n"
            f"If a gap cannot be auto-fixed:\n"
            f"  python3 .hooks/check_conformity_report.py "
            f"--disposition G-NN=exception\n\n"
            f"Full gap details:\n{body}"
        )

    auto_exhausted = (
        state["mode"] == "auto" and state["auto_iterations"] >= MAX_AUTO_ITERATIONS
    )
    mode_unknown = state["mode"] is None
    _block(_interactive_instruction(remaining, body, mode_unknown, auto_exhausted))


def _load_state_for_hash(current_hash):
    prev = read_state()
    if prev is None or prev["diff_hash"] != current_hash:
        state = fresh_state(current_hash)
        state["stop_hook_runs"] = prev.get("stop_hook_runs", 0) if prev else 0
    else:
        state = prev
        state.setdefault("stop_hook_runs", 0)
    state["stop_hook_runs"] += 1
    write_state(state)
    return state


def run_report():
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
        sys.exit(0)  # no code changes under src/

    state = _load_state_for_hash(current_hash)

    if state["status"] == "resolved":
        sys.exit(0)

    if state["stop_hook_runs"] > MAX_STOP_HOOK_RUNS:
        sys.exit(0)

    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", CODE_PATH],
        capture_output=True,
        text=True,
    ).stdout.strip()

    output, error = _run_auditor(changed)

    if error:
        _block(
            f"CONFORMITY HOOK — AUDITOR ERROR\n\n{error}\n\n"
            "Call AskUserQuestion: retry, proceed, or abort."
        )

    first_line, _, rest = output.partition("\n")
    body = rest.strip() if first_line.startswith("CONFORMITY_VERDICT:") else output

    if first_line.strip() == "CONFORMITY_VERDICT: PASS":
        state["status"] = "resolved"
        write_state(state)
        sys.exit(0)

    remaining = [
        (gid, title)
        for gid, title in parse_gaps(body)
        if gid not in state["dispositioned_gaps"]
    ]

    if not remaining:
        state["status"] = "resolved"
        write_state(state)
        sys.exit(0)

    _dispatch_gaps(state, remaining, body)


# ---------------------------------------------------------------------------
# State mutation subcommands
# ---------------------------------------------------------------------------


def run_set_mode(mode):
    if mode not in ("auto", "interactive"):
        print(f"Unknown mode '{mode}'. Use 'auto' or 'interactive'.", file=sys.stderr)
        return 1
    state = read_state()
    if state is None:
        print("No active conformity state. Trigger a Stop hook first.", file=sys.stderr)
        return 1
    state["mode"] = mode
    if mode == "auto":
        state["auto_iterations"] = 0
    write_state(state)
    print(f"Conformity mode set to '{mode}'.")
    return 0


def run_disposition(pairs_str):
    """Record gap dispositions. pairs_str: 'G-01=exception,G-02=skip'"""
    state = read_state()
    if state is None:
        print("No active conformity state.", file=sys.stderr)
        return 1
    recorded = []
    for pair in pairs_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        gid = pair.split("=")[0].strip()
        if gid and gid not in state["dispositioned_gaps"]:
            state["dispositioned_gaps"].append(gid)
            recorded.append(pair)
    write_state(state)
    if recorded:
        print(f"Recorded: {', '.join(recorded)}")
    return 0


# ---------------------------------------------------------------------------
# Pre-commit gate
# ---------------------------------------------------------------------------


def run_gate():
    current = diff_hash()
    if not current:
        return 0  # no code changes under src/ — nothing to gate

    state = read_state()
    if state is None or state["diff_hash"] != current:
        print(
            "No conformity check has been run for the current changes.\n"
            "Finish your Claude Code session — the Stop hook audits automatically."
        )
        return 1

    if state["status"] != "resolved":
        print(
            "Conformity gaps are not fully resolved.\n"
            "Return to your Claude Code session to complete the interactive review."
        )
        return 1

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = sys.argv[1:]

    if "--report" in args:
        run_report()
        return 0

    if "--set-mode" in args:
        idx = args.index("--set-mode")
        mode = args[idx + 1] if idx + 1 < len(args) else ""
        return run_set_mode(mode)

    if "--disposition" in args:
        idx = args.index("--disposition")
        pairs = args[idx + 1] if idx + 1 < len(args) else ""
        return run_disposition(pairs)

    return run_gate()


if __name__ == "__main__":
    sys.exit(main())

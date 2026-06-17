#!/usr/bin/env python3
"""Audit git history for commits missing required commit message elements.

Checks each commit for:
- Conventional Commits subject format (type(scope): description)
- Subject length ≤ 72 characters
- Non-empty body (why)
- Refs: footer

Usage:
    uv run python scripts/audit_commits.py
    uv run python scripts/audit_commits.py --since=v1.0.0
"""

import re
import subprocess
import sys
from dataclasses import dataclass, field

SUBJECT_PATTERN = re.compile(
    r"^(feat|fix|refactor|docs|test|chore|perf|ci)(\([a-z0-9/_-]+\))?: .+"
)


@dataclass
class CommitAudit:
    sha: str
    subject: str
    missing: list[str] = field(default_factory=list)


def _parse_log(since: str | None) -> list[tuple[str, str, str]]:
    """Return list of (sha, subject, full_body) tuples."""
    sep = "---COMMIT_SEP---"
    fmt = f"%H{sep}%s{sep}%b{sep}"
    cmd = ["git", "log", f"--format={fmt}"]
    if since:
        cmd.append(f"{since}..HEAD")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    commits = []
    for block in result.stdout.split(f"{sep}\n{sep}"):
        parts = block.split(sep)
        if len(parts) < 3:
            continue
        sha = parts[0].strip()
        subject = parts[1].strip()
        body = parts[2].strip() if len(parts) > 2 else ""
        if sha and subject:
            commits.append((sha[:12], subject, body))
    return commits


def audit(since: str | None = None) -> list[CommitAudit]:
    commits = _parse_log(since)
    results = []
    for sha, subject, body in commits:
        audit = CommitAudit(sha=sha, subject=subject)
        if not SUBJECT_PATTERN.match(subject):
            audit.missing.append("subject-format")
        if len(subject) > 72:
            audit.missing.append("subject-length")
        body_lines = [
            line
            for line in body.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not body_lines:
            audit.missing.append("body-why")
        if not any(line.startswith("Refs: ") for line in body.splitlines()):
            audit.missing.append("refs-footer")
        if audit.missing:
            results.append(audit)
    return results


def main() -> None:
    since = None
    for arg in sys.argv[1:]:
        if arg.startswith("--since="):
            since = arg.split("=", 1)[1]

    problems = audit(since)
    if not problems:
        print("All commits pass the commit message standard.")
        return

    print(f"Found {len(problems)} commit(s) with missing context:\n")
    print(f"{'SHA':<14} {'MISSING':<40} SUBJECT")
    print("-" * 90)
    for c in problems:
        missing_str = ", ".join(c.missing)
        subj = c.subject[:50] + ("…" if len(c.subject) > 50 else "")
        print(f"{c.sha:<14} {missing_str:<40} {subj}")

    print("\nLegend:")
    print("  subject-format  — does not match type(scope): description")
    print("  subject-length  — subject exceeds 72 characters")
    print("  body-why        — no body explaining why the change was needed")
    print("  refs-footer     — missing Refs: footer linking to issue/project")
    sys.exit(1)


if __name__ == "__main__":
    main()

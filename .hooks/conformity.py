#!/usr/bin/env python3
"""Shared helpers for the conformity-report hooks.

The conformity artifact (conformity.md) carries a machine-readable meta block:

    <!-- conformity-meta
    diff_hash: <sha256 of the code diff vs HEAD>
    state: pass | blocked | approved
    -->

Both the Stop hook (writer) and the pre-commit gate (reader) compute diff_hash the
same way, so a report is invalidated as soon as the code changes underneath it.
"""

import hashlib
import os
import re
import subprocess

REPORT_FILE = "conformity.md"
CODE_PATH = "src/"


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


_META_RE = re.compile(r"<!--\s*conformity-meta\s*(.*?)-->", re.DOTALL)


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

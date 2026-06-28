#!/usr/bin/env python3
"""Enforce that every commit references a specific GitHub issue."""

import re
import sys

ISSUE_RE = re.compile(r"^Refs: #\d+$")

lines = open(sys.argv[1]).read().splitlines()
if not any(ISSUE_RE.match(line) for line in lines):
    print("Commit message is missing a valid Refs: footer.")
    print("Every commit must reference an existing GitHub issue:")
    print("  Refs: #<issue-number>")
    print()
    print("If no issue exists yet, ask the Claude agent to create one first.")
    sys.exit(1)

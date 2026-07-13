#!/usr/bin/env python3
"""Enforce that a commit message has a non-empty body explaining why."""

import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()
non_comment = [line for line in lines if not line.startswith("#")]
body = "".join(non_comment[2:]).strip()  # skip subject line + blank separator
if not body:
    print("Commit body is empty.")
    print("Add a body explaining WHY this change is needed.")
    print("Explain the problem or constraint — not the mechanics.")
    print("Write for all readers: future-self, collaborators, AI agents, CI tooling.")
    sys.exit(1)

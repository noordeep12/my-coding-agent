#!/usr/bin/env python3
"""Enforce presence of a Refs: footer in commit messages."""

import sys

body = open(sys.argv[1]).read()
if not any(line.startswith("Refs: ") for line in body.splitlines()):
    print("Commit message is missing a Refs: footer.")
    print("Add a line such as:")
    print("  Refs: https://github.com/users/noordeep12/projects/1")
    print("  Refs: #<issue-number>")
    sys.exit(1)

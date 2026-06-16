#!/usr/bin/env python3
"""Enforce Conventional Commits subject format: type(scope): description."""
import re
import sys

subject = open(sys.argv[1]).readline().rstrip("\n")
pattern = r"^(feat|fix|refactor|docs|test|chore|perf|ci)(\([a-z0-9/_-]+\))?: .+"
if not re.match(pattern, subject):
    print("Commit subject does not follow Conventional Commits format.")
    print("Expected: type(scope): description")
    print("Types: feat fix refactor docs test chore perf ci")
    print(f"Got: {subject}")
    sys.exit(1)

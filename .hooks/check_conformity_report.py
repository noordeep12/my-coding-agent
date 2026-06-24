#!/usr/bin/env python3
"""Pre-commit gate: block commits whose code changes lack an approved conformity report.

The user always has the final word: a report in state 'blocked' stops the commit until
the user reviews conformity.md and sets state to 'approved'.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformity import REPORT_FILE, diff_hash, parse_report

current = diff_hash()
if not current:
    sys.exit(0)  # no code changes under src/ — nothing to gate

report = parse_report()
if report is None:
    print(f"No conformity report found. Expected {REPORT_FILE} for the changed code.")
    sys.exit(1)

if report["diff_hash"] != current:
    print(f"{REPORT_FILE} is stale: the code changed since the last conformity report.")
    print("Regenerate the report (the Stop hook does this at end of implementation).")
    sys.exit(1)

if report["state"] not in ("pass", "approved"):
    print(f"Conformity report state is '{report['state']}' — commit blocked.")
    print(f"Review {REPORT_FILE}; set state to 'approved' to give your final sign-off.")
    sys.exit(1)

sys.exit(0)

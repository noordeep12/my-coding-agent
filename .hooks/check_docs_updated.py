#!/usr/bin/env python3
"""Pre-commit hook: block commit if src/ changed but no docs were updated.

Docs that must be kept in sync with code changes:
  - README.md        (user-facing usage)
  - ARCHITECTURE.md  (structural decisions)
  - docs/            (Sphinx source)
"""

import subprocess
import sys

DOCS_PATHS = {"README.md", "ARCHITECTURE.md"}
DOCS_PREFIX = "docs/"
SRC_PREFIX = "src/"


def staged_files() -> list[str]:
    """Return staged file paths (added/copied/modified/renamed) for this commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],  # noqa: S607 -- relies on `git` from PATH, standard dev tooling
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def main() -> int:
    """Exit non-zero if src/ changed in this commit but no docs were updated."""
    files = staged_files()

    src_changed = any(f.startswith(SRC_PREFIX) for f in files)
    if not src_changed:
        return 0

    docs_changed = any(f in DOCS_PATHS or f.startswith(DOCS_PREFIX) for f in files)

    if not docs_changed:
        print(
            "ERROR: src/ files changed but no documentation was updated.\n"
            "\n"
            "When you change src/, update at least one of:\n"
            "  - README.md       (user-facing behaviour / usage)\n"
            "  - ARCHITECTURE.md (structural decisions)\n"
            "  - docs/           (Sphinx source)\n"
            "\n"
            "If this change genuinely requires no doc update, stage a\n"
            "no-op touch to one of the above files so the hook passes,\n"
            "and explain why in the commit body.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

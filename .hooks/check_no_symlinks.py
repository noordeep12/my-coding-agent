"""Reject staged symlinks (git mode 120000).

This repo has no legitimate symlinks. Agents working in git worktrees
have committed symlinks pointing at the main checkout's absolute path
(e.g. .claude, openspec in 874acda); valid where committed, they become
self-referential on checkout in the main repo and clobber the local
gitignored directory with an ELOOP.
"""

import subprocess
import sys


def main() -> int:
    """Fail if any staged entry is a symlink."""
    raw = subprocess.run(
        ["git", "diff", "--cached", "--raw", "--no-renames"],  # noqa: S607 -- relies on `git` from PATH, standard dev tooling
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    bad = [
        line.split("\t", 1)[1]
        for line in raw.splitlines()
        if line.split()[1] == "120000"
    ]
    if bad:
        print("Symlinks must not be committed to this repo:")
        for path in bad:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

---
name: run-quality-checks
description: Run this repo's tests, ruff, mypy, and security checks the way the Makefile and CI expect before declaring work done.
---

Before declaring a change complete, run the same gates CI and the pre-commit
hooks run, using the `Makefile` targets so flags match. Always drive Python
through `uv` — never the system `python`.

1. **Tests:** `make test` (`uv run pytest`). All tests must pass. If you added or
   changed `src/` behavior, add/extend tests in the same change.

2. **Lint + format:** `make lint` (`uv run ruff check src` and
   `uv run ruff format --check src`). To auto-fix, run `make format`
   (`ruff format src` then `ruff check --fix src`), then re-run `make lint`.

3. **Types:** `make typecheck` (`uv run mypy src`). Fix real type errors; do not
   silence them with blanket `# type: ignore`.

4. **Security:** `make security` (`uv run bandit -r src/ -ll` and
   `uv run pip-audit`).

5. **One shot:** `make check` runs test + lint + typecheck + security together —
   use it as the final gate.

Extra hooks fire at commit time and are worth running ahead of the commit if you
touched `src/`: the `docs-build` hook runs `sphinx-build -W` (warnings are
errors) and `docs-updated` requires a docs change in the same commit as any
`src/` change (see CONTRIBUTE.md). Green `make check` plus updated docs means the
work is actually done — report failures with their output rather than hiding them.

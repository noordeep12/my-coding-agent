# Contribute

## Mindset
### 1. Core Engineering Mindset
- Think as a **creator**, not a user.
- Always start with **WHY**: why does this system exist, and what problem does it solve?
- Then understand **HOW** it works internally and across systems.
- Be able to explain: problem → effort → solution clearly.
- Focus on design decisions; this is where engineering value exists (AI can implement, but context-driven design is human responsibility).

---

### 2. Problem Understanding (Before Coding)
Before implementation:
- What exact problem are we solving?
- Why does this problem exist?
- Who is affected?
- What outcome defines success?
- What assumptions are we making?

Do not start implementation without clarity on problem + assumptions.

---

### 3. Scope Control
- Keep changes small and focused.
- Avoid unrelated refactoring.
- Prefer incremental improvements over large rewrites.
- If scope grows, split into multiple tasks.

---

### 4. Complexity Control
- Minimize cognitive load in code.
- If code is hard to understand quickly, it is too complex.
- Prefer simple, explicit logic over clever implementations.
- Avoid unnecessary abstractions and hidden behavior.

---

### 5. Modularity & Information Hiding
- Design systems as independent modules.
- Each module must have a single responsibility.
- Hide internal complexity behind simple interfaces.
- Good modules do not require reading internal code to understand usage.

---

### 6. Change Isolation
- Prefer changes contained in a single file.
- Multi-file changes must be justified.
- Reduce cross-module coupling.
- Avoid widespread changes for small features.

---

### 7. Dependencies
- Minimize dependencies between components.
- Avoid circular dependencies completely.
- Prefer shallow dependency graphs.

---

### 8. Abstraction Rules
- Abstraction exists to reduce complexity, not increase it.
- Create abstractions only when multiple real use cases exist.
- Avoid pass-through methods.
- Generic components must solve real repeated problems.

---

### 9. Defaults & Edge Cases
- Design APIs so common cases are simple by default.
- Handle common edge cases internally where appropriate.
- Reduce special-case handling at call sites.

---

### 10. Data & State Management
- Keep state local when possible.
- Minimize variable usage scope.
- Avoid spreading state across many components.

---

### 11. Error Handling & Fail-Fast
- Fail-fast: detect and fail early when something is invalid.
- Do not ignore errors silently.
- Handle errors close to source when possible.
- Propagate only when caller can meaningfully act.
- Avoid unnecessary complexity in error chains.

---

### 12. Reliability Thinking
- Systems must work correctly even when things go wrong.
- Expect hardware failures, software bugs, and human mistakes.
- Design for fault tolerance and recovery.
- Test system behavior under failure conditions.

---

### 13. Scalability Thinking
When designing scalable systems:
- Define load parameters (requests/sec, reads/writes, concurrency).
- Identify performance bottlenecks (p50, p95, p99 latency).
- Choose between scaling up vs scaling out.
- Understand trade-offs: scalability increases complexity.

---

### 14. Maintainability
- Reduce long-term complexity.
- Favor designs that are easy to modify and debug.
- Hide backend complexity behind clean interfaces.
- Optimize for engineer productivity and system health.

---

### 15. Systems Thinking (Why / How)
- Always ask WHY a system is designed this way.
- Understand HOW systems work internally (e.g. network, storage, APIs).
- Know common system design patterns.
- Be able to reason about trade-offs and constraints.

---

### 16. Code Readability Rule
- Code must be understandable without jumping across multiple files.
- If understanding requires tracing many layers, the design is too complex.
- Prefer local reasoning over distributed reasoning.

---

### 17. Code Review Standards
All changes are reviewed for:
- Correctness
- Simplicity
- Security
- Edge cases
- Maintainability
- Hidden complexity

---

### 18. Testing Requirements
- Every feature must include tests.
- Bug fixes must include a regression test.
- Cover edge cases, not just happy paths.
- Tests validate behavior, not implementation details.

---

### 19. Observability
Production systems must be observable:
- Logs for debugging
- Metrics for performance and health
- Clear failure signals

If it runs in production, its behavior must be explainable.

---

### 20. Ownership
- You own your code after deployment.
- If it breaks, you are responsible for fixing it.

---

### 21. Security Baseline
- Validate all external inputs.
- Assume all external data is untrusted.
- Never expose secrets in logs or errors.
- Use least privilege access principles.

---

### 22. Study & Learning Principles
- Close material and summarize concepts from memory.
- Explain concepts simply (What / Why / How / Limits).
- Compare similar systems (e.g. replication vs partitioning).
- Study failure cases, not just ideal behavior.
- Read actively (ask why design decisions exist).

---

### 23. System Design Thinking Examples
- APIs exist for interoperability, reuse, and abstraction.
- Reliability = system continues working under faults.
- Scalability = ability to handle increasing load efficiently.
- Maintainability = ability to evolve system without excessive complexity.

Examples:
- Twitter fan-out: solve read-load scaling by precomputing timelines.
- LinkedIn profile as document: avoid expensive joins using document-based access.

---

### 24. Key Principle
Good engineering is about reducing complexity while increasing capability.
Bad engineering is hiding complexity without control.

---

## Python Development Standards

### 25. Project Structure

* Organize code by **feature/domain**, not file type, once the project is non-trivial.
* Keep a clear split: **public API → services/orchestration → core logic → adapters (IO, DB, APIs, LLMs)**.
* Ensure **one-way dependencies**; lower layers never import higher layers.
* Keep **core logic pure**, without IO, side effects, or external systems.
* Limit nesting depth (usually 3–4 levels max).
* Schemas live with the **domain they belong to**, not in a global `schemas/` folder, and should use typed models (Pydantic/dataclasses/type hints) as the single source of truth.
* A small, domain-local `utils.py` inside a package is fine for non-business helpers (response normalization, formatting). Avoid a global `utils/` or `helpers/` dump growing into unrelated code.
* **Never use `setup.py` for new projects.** All config lives in `pyproject.toml`.

#### Ideal package structure

Every domain package follows the same template, so the codebase is
consistent, straightforward, and free of surprises. Not every domain needs
every file — omit what a domain doesn't have, never add speculatively.

```
src/my_coding_agent/
├── <domain>/                  # one package per domain (engine, evals, webui, …)
│   ├── __init__.py            # public surface only — re-export the public API
│   ├── schema.py              # typed contracts: dataclasses, constants, type aliases
│   ├── exceptions.py          # domain exceptions inheriting MyCodingAgentError
│   ├── <feature>.py           # domain logic, one module per responsibility
│   │                          #   (runner.py, scoring.py — not a catch-all service.py)
│   ├── cli.py                 # entry surface if the domain has a CLI
│   │                          #   (thin: parse args → call domain function → render)
│   ├── server.py              # entry surface if the domain serves HTTP (thin handlers)
│   ├── store.py               # persistence adapter if the domain owns state (SQLite)
│   ├── utils.py               # optional: domain-local, non-business helpers
│   └── <subdomain>/           # nested package when the domain grows
│       ├── __init__.py        #   same template applies recursively
│       └── schema.py
├── utils/                     # generic cross-domain helpers only (lowest layer)
└── __init__.py

tests/
└── test_<domain>_<aspect>.py  # flat suite, named for the domain and behavior under test
```

Rules of thumb:
- A domain package owns everything about its domain: contracts, exceptions,
  logic, entry surfaces. New code goes into the domain it belongs to, not
  into a shared catch-all.
- Entry-surface modules (`cli.py`, `server.py`) stay thin — parse, validate,
  call the domain function, render/serialize. Logic never lives there.
- When a domain outgrows flat modules, split into subdomain packages
  (as `engine/llm/` and `engine/tool_execution/` do), each carrying its own
  `schema.py`.
- Existing packages are not restructured retroactively to match; the
  template governs new domains and new modules.

#### Layered dependency order

Arrange packages into responsibility layers and enforce strict one-way imports top-to-bottom:

```
generic helpers → passive capture → execution/domain → orchestration
```

Lower layers never import higher layers. If a layer must reference a higher layer at runtime, the import must be **lazy** (inside the function body, not at module level). Never suppress the linter warning without fixing the underlying coupling.

#### Cross-domain imports

When one domain package needs another, prefer keeping the **source module name
visible at the call site** — import the module (aliased if needed) rather than
bare symbols:

```python
# Good — origin is obvious at every call site
from my_coding_agent.evals import scoring as evals_scoring
from ..viewer import reader as viewer_reader

scorer = evals_scoring.resolve_scorer(scorer_ref)
sessions = viewer_reader.list_sessions(root)

# Avoid — at the call site, load_session reads like a local helper
from ..viewer.reader import load_session
```

Within a single domain package, importing bare symbols from sibling modules is
fine — this rule targets cross-domain boundaries, where knowing *which* domain
a function belongs to matters for reasoning about coupling. Apply it to new
code; do not churn existing imports solely to conform.

#### `schema.py` per module

Each package or subpackage that defines typed contracts (constants, type aliases, dataclasses, envelopes) collects them in a `schema.py`. Builder or executor logic stays in its own module. This keeps shape definitions discoverable and separates *what a thing looks like* from *how it works*.

#### `__init__.py` = public surface only

Re-export only the symbols that form the public API. Never pull underscore-prefixed private symbols up through `__init__.py` — even without `__all__`, they become importable and couple callers to internals. Tests that need private symbols import them directly from the submodule.

#### Passive vs active in observability

Observability code (recorders, event writers, metrics collectors) must only receive and record — it must never control execution flow. Active helpers that configure loggers, render output, or manage file handles belong in the utility layer, not in observability.

---

### 26. Tooling Requirements

| Tool | Purpose | Command |
|------|---------|---------|
| `uv` | Package management | `uv add`, `uv run` |
| `ruff` | Lint + format | `ruff check src && ruff format src` |
| `mypy` | Type checking | `mypy src` |
| `pytest` | Testing | `pytest --cov=my_coding_agent` |
| `bandit` | Security analysis | `bandit -r src/ -ll` |
| `pip-audit` | Dependency CVEs | `pip-audit` |

Run `make test && make lint` before every commit.

---

### 27. Code Quality Rules

#### Type Hints
- All public API functions must have type hints — no exceptions.
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (Python 3.10+).
- Include `py.typed` marker to signal typed package to downstream users.

```python
# Good
def process(items: list[str], timeout: int | None = None) -> dict[str, int]: ...

# Bad — no type hints
def process(items, timeout=None): ...
```

#### Anti-Patterns to Avoid

```python
# Bad: mutable default argument (shared across all calls — a classic bug)
def process(items: list = []):
    items.append(1)  # mutates the same list every time!

# Good: use None
def process(items: list | None = None):
    items = items or []

# Bad: bare except (swallows all errors including KeyboardInterrupt)
try:
    ...
except:
    pass

# Good: specific exception
try:
    ...
except ValueError as e:
    logger.error(e)

# Bad: boolean trap (caller can't tell what True/False means)
process(data, True, False, True)

# Good: keyword arguments
process(data, validate=True, cache=False, strict=True)
```

#### Pythonic Idioms

```python
# Iteration — never use range(len(...))
for item in items: ...
for i, item in enumerate(items): ...   # when index needed

# Dict access
value = d.get(key, default)            # not: if key in d: value = d[key]

# Context managers
with open(path) as f: ...              # not: f = open(); try/finally

# Comprehensions — only for simple, readable cases
squares = [x**2 for x in numbers]

# Set for O(1) membership checks
valid = set(allowed_values)
if item in valid: ...                  # not: if item in list(...)

# String concatenation — never use += in a loop
result = "".join(str(x) for x in items)
```

---

### 28. Naming Conventions

```python
# Actions: use verbs
encode(), decode(), validate(), process()

# Retrieval: get_ prefix
get_user(), get_config()

# Boolean: is_, has_, can_ prefix
is_valid(), has_permission(), can_retry()

# Conversion: to_ / from_ prefix
to_dict(), from_json()

# Private: underscore prefix
_internal_helper(), _cache
```

---

### 29. Error Handling Standards

Define a library-specific base exception. Give errors context and hints.
This project's base lives in `utils/exceptions.py`:

```python
class MyCodingAgentError(Exception):
    """Base exception — catch this to handle all library errors."""
    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint

# Usage
raise ValidationError(
    f"Latitude must be -90 to 90, got {lat}",
    hint="Did you swap latitude and longitude?"
)
```

- Fail-fast: validate inputs at entry points, not deep in the call chain.
- Never expose internal tracebacks or secrets in error messages.
- Handle errors close to the source; propagate only when the caller can act on it.

#### Domain exceptions live in their domain

Only truly cross-cutting errors (e.g. `PathTraversalError`) belong beside the
base in `utils/exceptions.py`. Domain-specific exceptions are defined in the
package that raises them and inherit the base, as `engine/llm/errors.py` and
`engine/checkpoint.py` already do:

```python
# engine/llm/errors.py
class LLMCallError(MyCodingAgentError): ...
class LLMTransportError(LLMCallError): ...
```

This keeps a module's failure modes discoverable next to its logic, and lets
callers catch either the narrow domain error or the project-wide base. New
domain exceptions must inherit `MyCodingAgentError`; standalone `Exception`
subclasses in domain packages are a red flag.

---

### 30. Testing Standards

#### Requirements
- Coverage target: **85% minimum** (configured via `--cov-fail-under=85`).
- Every public function needs tests; every bug fix needs a regression test.
- Cover edge cases: empty input, boundary values, invalid input, error paths.

#### Test Properties
| Property | Rule |
|----------|------|
| Independent | No shared mutable state between tests |
| Deterministic | Same result on every run, any environment |
| Fast | Unit tests must complete in < 100ms |
| Focused | Test behavior, not implementation details |

#### Patterns

```python
# Parametrize for multiple cases
@pytest.mark.parametrize("lat,lon,expected", [
    (37.7749, -122.4194, "9q8yy"),
    (90.0, 0.0, "zzzzzz"),          # boundary
])
def test_encode(lat, lon, expected):
    assert encode(lat, lon, precision=5) == expected

# Test exceptions explicitly
def test_invalid_lat_raises():
    with pytest.raises(ValueError, match="latitude"):
        encode(91.0, 0.0)

# Mock external dependencies — never hit real APIs in unit tests
def test_api_call(mocker):
    mocker.patch("my_lib.client.fetch", return_value={"data": []})
    assert my_lib.get_data() == []
```

---

### 31. Logging Standards

Library-layer code must **never configure logging** — that is the application's job.

```python
# Every module
import logging
logger = logging.getLogger(__name__)

# Package __init__.py — add NullHandler once
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Use logger, never print()
logger.debug("Processing %d items", len(items))

# NEVER in library-layer code
logging.basicConfig(...)   # configures root logger — breaks caller's logging
print("debug info")        # uncontrollable output
```

#### Application layer may configure logging

The rules above apply to library-layer packages (`engine/`, `pipeline/`,
`evals/`, `observability/`, `viewer/` internals). Application entry points —
the CLI commands (`src/my_coding_agent/cli.py`, `evals/cli.py`) and the HTTP
server (`viewer/server.py`) — own logging configuration and set it up
**once, at startup**, using the shared helpers in `utils/logging_core.py`.
Never configure logging as an import side effect.

---

### 32. Security Standards

```python
# SQL — always parameterized queries
conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))  # not: f"... {user_id}"

# Subprocess — never shell=True with user input
subprocess.run(["cat", filename], check=True)                 # not: shell=True

# Secrets — always from environment, never hardcoded
API_KEY = os.environ.get("API_KEY")

# Path traversal — always resolve and validate
base = Path("/data").resolve()
target = (base / user_input).resolve()
if not target.is_relative_to(base):
    raise ValueError("Path traversal detected")
```

Security checks run in CI on every PR: `bandit`, `pip-audit`, `detect-secrets`.

---

### 33. Dependencies

```toml
# Good: minimum version constraint
dependencies = ["requests>=2.28", "click>=8.0"]

# Bad: exact pin (locks users in, blocks security patches)
dependencies = ["requests==2.28.1"]

# Good: optional extras for non-core features
[project.optional-dependencies]
cli = ["click>=8.0"]
```

Minimize dependencies. Every dependency is a supply chain risk and a maintenance burden.

---

### 34. Performance Guidelines

Optimize only after profiling — never speculatively.

```bash
python -m pyinstrument script.py    # find the real bottleneck first
```

```python
# Use generators for large datasets — avoid loading everything into memory
def process(items):
    for item in items:
        yield transform(item)

# Cache expensive pure computations
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive(x: int) -> str:
    return compute(x)

# Use deque for queue operations (popleft is O(1), list.pop(0) is O(n))
from collections import deque
queue = deque()
queue.popleft()
```

Algorithm improvements first → data structure improvements → implementation tweaks.

---

### 35. HTTP API & Storage Conventions

The viewer exposes JSON APIs over stdlib `http.server`. Any storage layer
added for these or future JSON APIs follows the SQLite conventions below.

#### REST route conventions

- Name resources as **nouns**: `/api/sessions`, `/api/evaluations/<id>` — not
  `/api/get-sessions`.
- Use the **same path-variable name for the same concept** in every route:
  always `session_id`, never `sid` in one handler and `id` in another.
- Route handlers stay thin: parse and validate the request, call the domain
  function, serialize the response. Business logic lives in the domain
  package, never in the HTTP handler.

#### SQLite naming conventions

- `lower_snake_case` for all tables and columns.
- **Singular** table names for new tables (`eval_run`, not `eval_runs`).
  Existing tables (`items`) are grandfathered — do not rename retroactively.
- Group related tables with a domain prefix: `eval_run`, `eval_case`.
- `_at` suffix for datetime columns (`created_at`), `_date` suffix for dates.

#### SQL-first data shaping

- Push filtering, ordering, and aggregation into the SQL query; do not fetch
  all rows and post-process in Python loops.
- Build typed schema objects (dataclasses from the module's `schema.py`) from
  query results **at the boundary** — internals never pass raw `sqlite3.Row`
  objects around.

---

### 36. Red Flags in Python Code

If you see any of these, stop and fix them before merging:

- No `pyproject.toml` (using `setup.py` only)
- No type hints on public API
- No tests, or coverage below 70%
- Mutable default arguments
- Bare `except:` clauses
- `print()` statements in library code
- `logging.basicConfig()` in library code
- Hardcoded secrets or API keys
- `shell=True` in `subprocess` calls with user-controlled input
- Exact version pins in `dependencies` (use `>=` instead)
- Missing `LICENSE` file

---

### 37. API Design: The 90/10 Rule

Users employ ~10% of a library's functionality 90% of the time. Design accordingly:

- **Optimize the common case** — simple, zero-boilerplate path for standard operations.
- **Layer advanced options** — don't burden basic callers with advanced parameters.
- **Make wrong things hard** — use type hints and `Literal` types to restrict invalid inputs at IDE time, not just runtime.

```python
from typing import Literal

def encode(lat: float, lon: float, precision: Literal[1,2,3,4,5,6,7,8,9,10,11,12] = 12) -> str:
    ...
```

Error messages must guide toward the solution, not just describe the problem:

```python
# Bad
raise ValueError("Invalid latitude")

# Good
raise ValueError(
    f"Latitude must be between -90 and 90, got {lat}. "
    "Did you swap latitude and longitude?"
)
```

---

### 38. Code Complexity Control (McCabe)

**McCabe complexity** counts the number of independent execution paths through a function. High complexity = hard to test, hard to understand, high bug risk.

- Target: **≤ 3 per function** ideally; **≤ 10 project-wide maximum**.
- Ruff enforces this with `select = ["C901"]` and `max-complexity = 10`.

**Reduce complexity by:**

```python
# Bad: 3+ levels of nested conditionals
if user.is_active:
    if user.has_permission:
        if not user.is_blocked:
            return True

# Good: early returns (guard clauses)
if not user.is_active:
    return False
if not user.has_permission:
    return False
if user.is_blocked:
    return False
return True

# Bad: long if/elif chains
if action == "read":   handler = read_handler
elif action == "write": handler = write_handler
elif action == "delete": handler = delete_handler

# Good: lookup table (dict replaces conditionals)
HANDLERS = {"read": read_handler, "write": write_handler, "delete": delete_handler}
handler = HANDLERS.get(action)

# Good: use all()/any() instead of nested logic
if all([user.is_active, user.has_permission, not user.is_blocked]):
    return True
```

---

### 39. Docstring Standards

**Write docstrings first, before implementing the function.** Writing the docstring forces you to clarify what the function should do. If you can't write a clear docstring, the design is not ready.

```python
def encode(latitude: float, longitude: float, *, precision: int = 12) -> str:
    """Encode geographic coordinates to a geohash string.

    Args:
        latitude: Latitude in degrees, range -90 to 90.
        longitude: Longitude in degrees, range -180 to 180.
        precision: Output character count. Defaults to 12.

    Returns:
        Geohash string of the given precision.

    Raises:
        ValidationError: If coordinates are outside valid range.

    Example:
        >>> encode(37.7749, -122.4194)
        '9q8yy9h7wr3z'
    """
```

**Rules:**
- Use **imperative mood**: "Encode coordinates" not "Encodes coordinates".
- Every public function, class, and module must have a docstring.
- Pick **one style** (Google is recommended) and never mix styles in a codebase.
- Address: purpose, failure modes, edge cases, practical example.
- Use `functools.wraps` on decorators to preserve the wrapped function's docstring.

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)           # preserves func.__doc__ and __name__
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

---

### 40. Documentation as a Quality Gate

Documentation is not optional — it is the difference between a library that gets used and one that gets ignored.

- **Treat Sphinx warnings as errors** — run `sphinx-build -W` in CI; broken links and missing docstrings fail the build.
- **Run docs build on every PR** — documentation issues are bugs; catch them before merge.
- **Read your own docs** — after building, read through the API reference as if you were a new user. If you can't find something, fix the navigation.
- The success test: a user can find information, understand the problem, understand usage, and implement functionality **without asking you questions**.

---

### 41. Documentation Update Policy

Every change to `src/` **must** be accompanied by a documentation update in the same commit. This is enforced by the `docs-updated` pre-commit hook.

#### What counts as a documentation update

| What changed | Minimum doc to update |
|---|---|
| New public function / class / module | Docstring + `docs/` API reference |
| Behaviour or interface change | `README.md` usage section |
| Architectural decision or new component | `ARCHITECTURE.md` |
| Config or CLI flag change | `README.md` |
| Internal refactor with no observable change | `ARCHITECTURE.md` (note the restructure) |

#### Files the hook monitors

- `README.md` — user-facing usage and behaviour
- `ARCHITECTURE.md` — structural decisions and component layout
- `docs/` — Sphinx source (API reference, guides)

#### If a change genuinely needs no doc update

Stage a no-op touch to the most relevant file and explain **why** in the commit body. The hook checks presence, not content depth — the commit body carries the justification.

```bash
# Example: internal test helper refactor with no observable change
touch ARCHITECTURE.md
git add ARCHITECTURE.md
# commit body: "Internal helper extraction in tests/; no public API or
# structure change. ARCHITECTURE.md touched to satisfy docs-updated hook."
```

#### Why this matters

Documentation debt accumulates silently. By the time it is noticed it is expensive to reconstruct — especially for architectural decisions where the original reasoning is lost. Keeping docs in sync at commit time costs seconds; reconstructing them later costs hours.

---

### 42. Security: Additional Patterns

#### Log Injection Prevention

f-strings preserve newlines — an attacker can forge log entries by injecting `\n` into user input.

```python
# Bad: f-string in logging lets newlines forge log entries
logger.info(f"User {user_id} performed: {action}")

# Good: use % formatting — logging sanitizes the interpolation
logger.info("User %s performed: %s", user_id, action)
```

#### YAML Loading

```python
# Bad: yaml.load() can execute arbitrary Python via YAML tags
data = yaml.load(user_content)

# Good: always safe_load()
data = yaml.safe_load(user_content)
```

#### Input Validation Order

Validate sequentially at every entry point — external inputs are untrusted by definition:

```
1. Type check (isinstance)
2. Length check (prevent DoS via huge inputs)
3. Format check (regex, enum membership)
4. Business rule check (range, relationship constraints)
```

Use allowlists (specify what is valid) rather than denylists (block known bad patterns).

#### Secrets in Objects

Override `__repr__` and `__str__` to prevent accidental secret exposure in logs and tracebacks:

```python
class ApiClient:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def __repr__(self) -> str:
        return f"ApiClient(api_key='****')"   # never expose key

    def __str__(self) -> str:
        return self.__repr__()
```

Provide factory methods rather than accepting raw secrets as arguments:

```python
@classmethod
def from_env(cls) -> "ApiClient":
    key = os.environ.get("API_KEY")
    if not key:
        raise ValueError("API_KEY environment variable not set")
    return cls(key)
```

#### Secure Temporary Files

```python
import tempfile

# Bad: predictable path, race condition, accessible to others
tmp = "/tmp/myfile"

# Good: unique, mode-600, auto-cleaned
with tempfile.TemporaryDirectory() as tmp_dir:
    ...
```

---

### 43. Testing: Coverage and Multi-Environment

#### Coverage: Quality over Quantity

High coverage percentage does not guarantee quality tests. You can execute code without asserting correct behavior — coverage shows *what* ran, not *whether it was correct*.

- Enable **branch coverage** (not just line coverage) — branch coverage catches missing `else` paths.
- Target: **85–95% on critical components**; don't chase 100% (diminishing returns on trivial code).
- Dead code detected by coverage tools should be removed, not excluded.

```toml
# This project's actual configuration (pyproject.toml)
[tool.pytest.ini_options]
addopts = "--cov=my_coding_agent --cov-branch --cov-report=term-missing --cov-fail-under=85"

[tool.coverage.run]
branch = true
source = ["src/my_coding_agent"]

[tool.coverage.report]
show_missing = true
omit = ["*/schema.py"]
```

#### Multi-Environment Testing with Tox

Your local environment is one of thousands of environments your library will run in. Test against all supported Python versions and both minimum and latest dependency versions.

```toml
# pyproject.toml
[tool.tox]
legacy_tox_ini = """
[tox]
envlist = py{310,311,312}-deps{min,latest}

[testenv]
deps =
    deps-min: requests==2.28.0    # test minimum supported
    deps-latest: requests         # test latest
commands = pytest
"""
```

#### Mocking: Use Strategically, Not Excessively

- The goal is **reliable, maintainable tests** — not maximum mock count.
- Mock external dependencies (HTTP, databases, filesystems, clocks) that are slow, expensive, or unreliable.
- Do not mock your own code's internal logic — that tests the mock, not the code.
- Use `mocker` fixture from `pytest-mock` (auto-cleans up after each test, unlike `unittest.mock.patch` decorators).

```python
# Useful mocking helpers beyond mocker.patch:
# freezegun — freeze time for time-dependent code
# responses  — mock HTTP requests cleanly
# moto       — mock AWS services without real calls
```

---

### 44. Performance: Benchmark → Profile → Optimize

The correct order is always:

```
1. Benchmark  →  confirm there is a real, measurable problem
2. Profile    →  find exactly WHERE time or memory is spent
3. Optimize   →  change only the confirmed bottleneck
4. Benchmark  →  verify the improvement
```

**Benchmark** measures total execution time. **Profiling** explains why.

- Use `pytest-benchmark` to detect regressions across commits:
  ```bash
  pytest --benchmark-save=baseline      # save reference
  pytest --benchmark-compare=baseline   # flag slowdowns
  ```

- **CPU profiling ≠ memory profiling.** A function can be fast but allocate enormous memory (or vice versa). Profile both independently.
  - CPU: `python -m pyinstrument script.py` (statistical, low overhead)
  - Memory: `memray run script.py` (tracks C extension allocations too — standard Python profilers miss these)

Never optimize speculatively. Never rely on intuition about which code is slow.

---

### 45. pyproject.toml: Why It Replaced setup.py

`setup.py` ran arbitrary Python code during installation — a security risk and a source of fragile bootstrapping bugs. `pyproject.toml` is declarative (states *what* the project needs, not *how* to build it), which is:

- **Safer** — no code execution during install
- **Reproducible** — tools read static configuration, no side effects
- **Unified** — one file replaces `setup.py`, `setup.cfg`, `MANIFEST.in`, `requirements.txt`, and per-tool config files

Key PEPs behind modern packaging:
- PEP 518 — build requirements (`[build-system]`)
- PEP 621 — project metadata (`[project]`)
- PEP 660 — editable installs (`pip install -e .`)

---

### 46. Commit Standards

Every commit must answer four questions so any reader — future-self, collaborators,
AI agents, CI tooling — can understand it in isolation without tracing code.

| Question | Where | Enforced |
|----------|-------|---------|
| **What** changed | Subject: `type(scope): description` | `commit-subject-format`, `commit-subject-length` |
| **Why** it was needed | Body: non-empty explanation of the problem | `commit-body-required` |
| **For whom** it matters | Implicit in a complete body written for all readers | — (style, not a separate field) |
| **Which issue** it addresses | Footer: `Refs: #<issue>` | `commit-refs-footer` |

**Subject rules:**
- Use **Conventional Commits**: `type(scope): description`
- **≤ 72 characters**, imperative present tense
- **Types:** `feat` `fix` `refactor` `docs` `test` `chore` `perf` `ci`

**Body rules:**
- Must be non-empty — explain the *problem or constraint* that motivated the change, not the mechanics
- Write as if the reader has no other context: future-self after 6 months, a collaborator, an AI agent parsing history
- Wrap lines at ~72 chars

**Footer rules:**
- Must include `Refs: #<issue-number>` referencing an **existing** GitHub issue
- If no issue exists for the change, the Claude agent must create one before committing

All four constraints are enforced locally by pre-commit hooks at `commit-msg` stage
(`.pre-commit-config.yaml`). A commit missing any element is rejected before it lands.

A commit-message template lives at the repository root (`.gitmessage`) and models
this convention. Enable it locally:

```bash
git config commit.template .gitmessage
```

---

### 47. Branch & Pull Request Standards

- Every change reaches `main` through a pull request — **never a direct push**.
- The PR title becomes the squash-merge commit subject, so it follows the §46
  subject rules: `type(scope): description`, ≤ 72 characters. Enforced by
  `.github/workflows/pr-title.yml`.
- **One concern per PR.** Guideline: ~200–400 changed lines; beyond that, split.
  Small PRs are reviewed faster, revert cleanly, and hide fewer bugs (Google
  eng-practices, "Small CLs") — doubly important when the author is an
  unattended agent.
- Delete the branch after merge; a merged branch left behind is repo debt (§50).

---

### 48. Enforcement Parity

**Every mechanically checkable rule in this document MUST be enforced by an
enabled tool rule.** A standard that lives only in prose decays silently; a
check that lives only in tooling (with no documented rule) cannot be audited.
Both directions are gaps.

Required mapping between documented rules and enforcement:

| Documented rule | Enforcing check |
|-----------------|-----------------|
| §27 anti-patterns (mutable defaults, bare except, boolean traps) | ruff `B` (bugbear) |
| §27 Pythonic idioms, modern syntax | ruff `UP`, `SIM`, `RUF` |
| §30 test patterns | ruff `PT` |
| §32/§42 security patterns | ruff `S` + bandit |
| §39 docstring standards (Google style, every public symbol) | ruff `D` with `convention = "google"` |
| §50 dead code | ruff `ERA`, `ARG` |
| §46/§47 commit and PR subjects | commit-msg hooks + `pr-title.yml` |

- mypy `strict = true` is the target configuration. Each per-module override is
  tracked debt and requires an open issue; the override list only shrinks.
- Enforcement thresholds (`--cov-fail-under`, enabled ruff families, mypy
  overrides) are **ratchets**: they may only tighten. A loosened threshold is a
  standards violation, not a configuration choice.

---

### 49. Versioning & Release

- Versions follow **Semantic Versioning** (semver.org): breaking / feature /
  fix maps to major / minor / patch.
- `CHANGELOG.md` follows the **Keep a Changelog** format with an `Unreleased`
  section. Every user-facing change updates it **in the same PR**.
- The git tag `vX.Y.Z` must equal `[project] version` in `pyproject.toml` at
  the tagged commit. A mismatch between tag, version, and changelog is a
  release-process violation.
- Every release is an annotated tag plus a GitHub Release whose notes come from
  the changelog entry. No untagged version bumps; no unreleased tags.

---

### 50. Dead Code & Repo Hygiene

- **No exported-but-unreferenced public surface.** A symbol re-exported in
  `__init__.py` or kept on a module's public API with no caller and no external
  contract must be removed, not retained "just in case" (see §8: build
  abstractions when needed, not speculatively).
- Dead code is **removed**, never commented out and never excluded from
  coverage to hide it (extends §43).
- `[project] dependencies` contains only packages the code imports; unused
  dependencies are removed as soon as detected.
- Workflow artifacts (`PROBLEM.*.md`, `gap.md`, generated reports) live in
  dedicated directories, not the repository root — the root listing is context
  every agent session pays for. The dedicated directory is `.workbench/`
  (gitignored).
- Merged branches are deleted (§47); stale local/remote branches, caches, and
  build artifacts are pruned rather than accumulated.

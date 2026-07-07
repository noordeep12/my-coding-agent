# Security

## The dangerous-command refusal gate

Every `bash` tool call the model emits is checked against a small, deterministic
rule set — **before** it runs — in
[`engine/tool_execution/policy.py`](src/my_coding_agent/engine/tool_execution/policy.py).
A command that matches a rule (e.g. `rm -rf /`, `curl … | sh`, `dd of=/dev/sda`)
never reaches the shell. The model gets back a structured refusal instead: what
was refused, why, a reference to a recognized security standard (CWE/OWASP/NIST),
and a safer alternative — so it can steer to a working approach instead of
retrying blind.

This is **defense-in-depth, not a sandbox**. It's a last line of defense against
unambiguous, high-signal danger — not protection against a determined adversary
or an obfuscated command (base64, `$IFS`, `eval` tricks are not de-obfuscated).
The existing system-prompt safety guidance still matters; this gate backs it up
with something actually enforced.

Every refusal is recorded (`events.jsonl`, a `refusal` event) and logged
(`stderr.log`, a WARNING line), and shows up in the [Trace Explorer](README.md#trace-explorer)
with a dedicated badge and detail panel — so you can audit what was blocked and
why after the fact.

---

## Disabling the gate

**Off by default is the wrong instinct here — think twice before disabling this.**
It exists because an unattended agent with an unenforced "please don't"
in the system prompt is one hallucinated command away from real damage. If
you're disabling it because a rule is misfiring on something you need to do
legitimately, prefer fixing the rule (see below) over turning the whole gate off.

That said, sometimes you need it off — a sandboxed CI container, a throwaway VM,
or debugging the gate itself. Two equivalent ways, same effect for the process:

**CLI flag** (most discoverable):
```bash
uv run my-coding-agent --no-safety-gate --prompt "..."
```
Prints a loud warning to stderr every time it's used, so it's never silently on.

**Environment variable** (for scripts/CI, or if you're not going through the CLI):
```bash
export MCA_DISABLE_DANGEROUS_COMMAND_GATE=1
uv run my-coding-agent --prompt "..."
```
Any value other than empty, `0`, or `false` disables the gate. The CLI flag is
just this same variable, set for you.

There is no per-rule opt-out (e.g. "disable only the git-force-push rule") — it's
all-or-nothing by design, to keep the gate simple to reason about and audit. If
you need that, extend the rule instead (next section).

---

## Extending the rule set

Adding a rule is a two-step, no-plumbing change: write a predicate function,
then register it. You never touch `ToolExecutor`, the recorder, or the viewer —
they already handle any rule in `RULES` generically.

### 1. Write the predicate

A predicate is a plain function: `(command: str) -> bool`. Return `True` only
when you're confident the command matches the dangerous pattern — a false
positive blocks legitimate work and erodes trust in the gate, so **when in
doubt, return `False`** (the opposite bias of "detect everything").

```python
# Somewhere near the other predicates in policy.py:

_MY_DANGEROUS_TOOL_RE = re.compile(r"\bsome-dangerous-cli\b.*--force\b")


def _is_my_dangerous_pattern(command: str) -> bool:
    return bool(_MY_DANGEROUS_TOOL_RE.search(command))
```

Reuse `shlex.split()` if you need to inspect individual words/flags rather than
regex-match the raw string — see `_is_destructive_git_push` for an example that
parses out flags precisely (so `--force-with-lease` doesn't false-positive on
a plain `--force` regex).

### 2. Register a `Rule`

Add an entry to the `RULES` tuple with:

| Field | What it is | Example |
|---|---|---|
| `rule_id` | Stable snake_case id — shows up in `events.jsonl`, the Trace Explorer, and tests | `"my_dangerous_pattern"` |
| `predicate` | The function from step 1 | `_is_my_dangerous_pattern` |
| `reason` | One or two plain-English sentences: what makes this dangerous | see examples in `policy.py` |
| `references` | A tuple of `Reference(standard_id, url)` — **at least one**, a real CWE/OWASP/NIST citation, not a placeholder | `(_CWE_78,)` or a new `Reference(...)` |
| `safer_alternative` | Concrete guidance the model can act on | `"Use --dry-run first, or pass an explicit target instead of a glob."` |

```python
RULES: tuple[Rule, ...] = (
    # ...existing rules...
    Rule(
        rule_id="my_dangerous_pattern",
        predicate=_is_my_dangerous_pattern,
        reason=(
            "some-dangerous-cli --force skips confirmation and can overwrite "
            "production state with no way to undo it."
        ),
        references=(_CWE_78,),  # or define a new Reference(...) near the others
        safer_alternative=(
            "Run without --force first to preview the change, or use "
            "--dry-run to confirm the target before forcing."
        ),
    ),
)
```

That's it. `ToolExecutor.invoke_tool` already calls `policy.evaluate()` for
every `bash` call, `after_tool_call` already builds the refusal envelope from
whatever `Rule` matched, and the recorder/viewer already know how to show it —
nothing else needs to change.

### 3. Test it

Add your command to the parametrized tests in
[`tests/test_tool_policy.py`](tests/test_tool_policy.py):

```python
# In test_evaluate_refuses_dangerous_commands's parametrize list:
"some-dangerous-cli --force target",

# In test_evaluate_allows_safe_look_alikes's parametrize list — a close-but-safe
# look-alike your predicate must NOT fire on:
"some-dangerous-cli --dry-run target",
```

`test_every_rule_carries_reason_reference_and_safer_alternative` and
`test_refusal_exposes_same_fields_as_its_rule` already cover every rule in
`RULES` generically — you don't need to duplicate those for your new rule.

Run just the policy tests while iterating:

```bash
uv run pytest tests/test_tool_policy.py -v
```

### What NOT to do

- **Don't** make a rule fire on a broad pattern "to be safe" — a rule that
  blocks legitimate work trains users to reach for `--no-safety-gate`, which
  defeats the entire point. Narrow and high-signal beats broad and paranoid.
- **Don't** add a rule for a non-`bash` tool yet — the interface
  (`evaluate(func_name, args)`) is ready for it, but no other tool is gated
  today; that's a separate, larger change (registry-level gating, not a
  `policy.py`-only addition).
- **Don't** try to catch obfuscation (base64-encoded commands, `$IFS` tricks,
  `eval "$(...)"`). This is a documented, accepted limitation — a textual gate
  can't reliably de-obfuscate, and chasing it invites false positives without
  closing the gap completely. If you're worried about a determined attacker,
  you want an actual sandbox, not a smarter regex here.

---

## The network egress filter

Every `fetch_web` call is checked against an actively-maintained, open-source
blocklist of publicly-catalogued malicious domains — **before** the connection
proceeds — in
[`engine/egress/`](src/my_coding_agent/engine/egress/__init__.py). A
destination whose host (or a parent domain) is on the list never reaches
`httpx`. The model gets back a structured block instead: which host, and which
blocklist it matched — so it can steer to a legitimate destination.

The default posture is **deny known-bad, allow unknown**: a host absent from
the blocklist proceeds exactly as before. This is a known-bad layer on top of
whatever default-deny sandboxing exists elsewhere, not a novel-threat
detector — a host the security community hasn't catalogued yet is not caught.

The blocklist (primary source: [hagezi Threat-Intelligence-Feeds](https://github.com/hagezi/dns-blocklists);
secondary: [abuse.ch URLhaus](https://urlhaus.abuse.ch/)) is fetched to a local
cache (`~/.my_coding_agent/egress_cache/blocklist.txt` by default) and
refreshed once the cache is more than 24 hours old. It is **offline-tolerant
by design**: a refresh that fails falls back to the last-good cache, and a run
with no cache at all and no working refresh proceeds with the filter *open*
(a logged warning, not a hard failure) — a stale or unreachable blocklist
never blocks all outbound work.

Every block is recorded (`events.jsonl`, an `egress` event) and logged
(`stderr.log`, a WARNING line), and shows up in the [Trace Explorer](README.md#trace-explorer)
as an errored `fetch_web` tool dispatch with the host and matched list in its
metadata.

### Disabling the filter

**CLI flag:**
```bash
uv run my-coding-agent --no-egress-filter --prompt "..."
```
Prints a loud warning to stderr every time it's used.

**Environment variable:**
```bash
export MCA_DISABLE_EGRESS_FILTER=1
uv run my-coding-agent --prompt "..."
```
Any value other than empty, `0`, or `false` disables the filter. When
disabled, `fetch_web` behaves exactly as it did before this feature existed,
and no `egress` events are recorded.

**Selecting the blocklist source** (`hagezi` by default):
```bash
export MCA_EGRESS_FILTER_SOURCE=urlhaus
```

---

## Reporting a security issue

If you find a way to bypass the gate that you think is worth fixing broadly
(not just "add one more rule"), open an issue describing the bypass and, if
possible, a representative command that gets through undetected.

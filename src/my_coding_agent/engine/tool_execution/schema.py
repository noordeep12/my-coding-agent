"""Canonical tool-output schema — the agent-facing result contract.

Every tool result that reaches the agent — including auto-triggered paths
(artifact summaries, skips, parse/arg errors, raised exceptions) — is normalized
into one envelope so success/failure is uniform and machine-checkable, modeled on
bash's ``ok``/``exit_code``. ``output`` carries the raw payload (stdout / file
content / report / summary); tool-specific extras go in the flexible ``metadata``
bag.

This module holds shapes only — the schema version, envelope key set, preview
descriptor keys, and the error-string pattern. The builders that assemble and
validate the envelope live in the sibling :mod:`envelope` module.
"""

import re

TOOL_SCHEMA_VERSION = 1
_TOOL_RESULT_KEYS = ("schema_version", "tool", "ok", "output", "error", "metadata")
_ERROR_PREFIX_RE = re.compile(r"^Error\b")

# Shape of ``metadata.preview`` for an offloaded artifact: a descriptor of the
# bounded excerpt carried in ``output``. The excerpt itself and the skim guidance
# live in ``output`` (not here); the full raw output lives only on disk.
PREVIEW_KEYS = (
    "shown_lines",
    "total_lines",
    "shown_bytes",
    "total_bytes",
    "full_output_path",
)

# Shape of one entry in ``metadata.duplicate_of`` (keyed by stream: "stdout"/
# "stderr"): locates a deduplicated read-back within the artifact it duplicates.
DUPLICATE_OF_KEYS = ("tool_call_id", "stream", "offset", "length")

# ── Size/threshold configuration ──────────────────────────────────────────────
# Single source of truth for every tunable boundary that decides whether/how a
# tool output is offloaded to the on-disk artifact store. Both tool_execution
# and tool_registry read these — this module is a leaf (no internal imports),
# so either side can depend on it without introducing a cycle.

_CHARS_PER_TOKEN = 4  # rough chars/token estimate used only for budgeting

# Large-output boundary (~10,000 tokens): a tool's raw output above this size
# takes the ``(None, artifact_dict)`` offload contract instead of returning inline.
ARTIFACT_THRESHOLD_TOKEN_BUDGET = 10_000
ARTIFACT_THRESHOLD = ARTIFACT_THRESHOLD_TOKEN_BUDGET * _CHARS_PER_TOKEN

# Alias: the same boundary, named for its use truncating a plain-string
# (non-artifact) tool output that exceeds it.
MAX_TOOL_OUTPUT_CHARS = ARTIFACT_THRESHOLD

# Sanity cap on a fetched page's body: guards a pathological page. A
# separate, much larger boundary than ARTIFACT_THRESHOLD by design — everything
# under it still gets evaluated for offload normally (fidelity within the cap
# is preserved on disk via offload), it just stops a runaway fetch from being
# considered at all. Expressed as a multiple of ARTIFACT_THRESHOLD_TOKEN_BUDGET
# so the two move together if the offload boundary changes.
PAGE_FETCH_MAX_TOKEN_BUDGET = ARTIFACT_THRESHOLD_TOKEN_BUDGET * 5
PAGE_FETCH_MAX_CHARS = PAGE_FETCH_MAX_TOKEN_BUDGET * _CHARS_PER_TOKEN

# Preview budget for an offloaded artifact: only a bounded excerpt goes into the
# tool result `output`; the full content stays on disk. Deliberately a small
# fraction of ARTIFACT_THRESHOLD_TOKEN_BUDGET so the preview never itself
# approaches the offload boundary.
PREVIEW_TOKEN_BUDGET = ARTIFACT_THRESHOLD_TOKEN_BUDGET // 10  # ~1000 tokens
PREVIEW_MAX_CHARS = PREVIEW_TOKEN_BUDGET * _CHARS_PER_TOKEN

# Per-call cap for a `read_tool_artifact` byte-range slice. Aliased to
# PREVIEW_MAX_CHARS so the boundary is named where the other offload constants
# live, rather than introducing an independent budget.
RANGE_MAX_CHARS = PREVIEW_MAX_CHARS

# Output cap for a single `read_tool_artifact(query=...)` extraction call — the
# separate LLM call that scans the artifact for a query-scoped answer. Lives
# here (not tool_registry/registry.py, the caller) so the offload-preview
# guidance in tool_execution/output.py can disclose the same number the tool
# actually enforces, without duplicating or drifting from it.
EXTRACTION_OUTPUT_TOKEN_BUDGET = 800  # bounds a single read_tool_artifact return
EXTRACTION_OUTPUT_MAX_CHARS = EXTRACTION_OUTPUT_TOKEN_BUDGET * _CHARS_PER_TOKEN

# ── Skill-index budget (skill-knowledge-delivery, issue #19) ───────────────────
# The skill index is appended to the session's opening user message (never the
# system prompt, so the #75 prefix-cache invariant holds). These fixed character
# caps bound it deterministically — no LLM, no ranking. Kept here alongside the
# other offload/size thresholds so every size boundary lives in one module.

# Per-entry cap: the maximum length of one rendered index line
# (``- <name>: <description>``); a longer description is truncated with an
# ellipsis.
SKILL_INDEX_PER_ENTRY_MAX_CHARS = 200

# Total cap: the rendered index (header + all lines) never exceeds this. Over
# budget, descriptions are truncated evenly; still over, lines degrade to
# names-only. The block placed into the opening message never exceeds this cap.
SKILL_INDEX_TOTAL_MAX_CHARS = 2_000

# ── Tool-result supersession (issue #121) ─────────────────────────────────────
# The machine-written marker `read_tool_artifact`'s query mode appends to a
# return when it cut content (see `tool_registry/registry.py`'s
# `_extraction_disclosure`). Shared here so the supersession pass (Case A) can
# detect it without importing the registry module back.
EXTRACTION_INCOMPLETE_MARKER = "[Extract incomplete —"

# Minimum retired-content size below which a provably-superseded tool result is
# left alone: replacing a message invalidates the KV prefix cache from that
# point forward, so the one-time re-prefill cost must be smaller than the
# repeated re-pay a retirement avoids. Picked below the ~3,200-char incomplete-
# extract fragment measured in the motivating session (issue #121,
# ebd9ae84480d) so that case is covered, while trivial results are never worth
# the churn.
SUPERSESSION_SIZE_FLOOR_CHARS = 500

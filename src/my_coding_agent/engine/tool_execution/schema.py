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

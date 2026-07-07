"""Secret-exfiltration guard — deterministic, local, point-of-egress check.

This module is a stdlib-only leaf (no internal imports), the exfil analog of
``tool_execution.policy``: rules are plain regexes, matched deterministically
against a path or an outbound payload. :func:`is_sensitive` and
:func:`scan_payload` are the two entry points a tool wires in before letting
content leave the machine. Neither function ever returns the matched value —
only the category name — so a caller cannot accidentally echo the secret
itself into an error message or an event row (design.md decision 3).

Reads are never blocked here; this module only answers "is this sensitive",
the caller (e.g. ``fetch_web``) decides to block at the point of egress.
Deliberately high-signal and narrow, like ``policy``: a false positive blocks
legitimate work, so only well-known patterns fire. Not exhaustive — encoded/
obfuscated secrets (base64, split across params) defeat a textual scan by
design; that is an acknowledged limitation (design.md risks), not a bug.
"""

from __future__ import annotations

import os
import re

from .schema import (
    CATEGORY_CLOUD_CREDENTIALS,
    CATEGORY_ENV_FILE,
    CATEGORY_KEY_FILE,
    CATEGORY_NETRC,
    CATEGORY_PRIVATE_KEY_CONTENT,
    CATEGORY_SECRET_NAMED_PATH,
    CATEGORY_SSH_KEY,
    CATEGORY_TOKEN_CONTENT,
)

# Opt-out switch (off by default): set to any value other than ""/"0"/"false"
# to disable the guard for the process — mirrors
# ``tool_execution.policy.DISABLE_ENV_VAR``. Read at call time, not import
# time, so a CLI flag setting this env var before the first tool call and a
# shell-exported var behave identically.
DISABLE_ENV_VAR = "MCA_DISABLE_EXFIL_GUARD"


def is_guard_disabled() -> bool:
    """Return ``True`` when the secret-exfiltration guard is disabled."""
    raw = os.environ.get(DISABLE_ENV_VAR, "")
    return raw.strip().lower() not in ("", "0", "false")


# ── well-known sensitive filename/path patterns ────────────────────────────
# Aligned with the project's own never-commit list (CLAUDE.md Red Flags):
# .env*, .ssh/id_*, .aws/credentials, .netrc, *.pem/*.key/*.p12,
# *secret*/*credential*, keychain exports.
_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (CATEGORY_ENV_FILE, re.compile(r"(^|[/\\])\.env(\.\w+)?(?:[/\\]|$)")),
    (CATEGORY_SSH_KEY, re.compile(r"\.ssh[/\\]id_\w+")),
    (CATEGORY_CLOUD_CREDENTIALS, re.compile(r"\.aws[/\\]credentials\b")),
    (CATEGORY_NETRC, re.compile(r"(^|[/\\])\.netrc(?:[/\\]|$)")),
    (CATEGORY_KEY_FILE, re.compile(r"\.(pem|key|p12)\b")),
    (
        CATEGORY_SECRET_NAMED_PATH,
        re.compile(r"(secret|credential)", re.IGNORECASE),
    ),
)

# ── content signature patterns ─────────────────────────────────────────────
# Raw (unencoded) secret shapes — PEM private-key headers and common token
# formats. Does not attempt to catch encoded/obfuscated secrets (documented
# limitation, design.md risks).
_CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        CATEGORY_PRIVATE_KEY_CONTENT,
        re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    ),
    (CATEGORY_TOKEN_CONTENT, re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    (CATEGORY_TOKEN_CONTENT, re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (CATEGORY_TOKEN_CONTENT, re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
)


def is_sensitive(path: str) -> str | None:
    """Return the matched sensitivity category for ``path``, or ``None``.

    Deterministic filename/path matching only — no file I/O, no content
    inspection. Never returns the path itself, only the category name.
    """
    for category, pattern in _PATH_PATTERNS:
        if pattern.search(path):
            return category
    return None


def scan_payload(data: str) -> str | None:
    """Return the matched sensitivity category for ``data``, or ``None``.

    Checks content signatures (private-key headers, common token formats)
    first, then whether ``data`` references a known-sensitive path (e.g. a
    URL or command line naming ``~/.ssh/id_ed25519``). Deterministic and
    local: no LLM call, no network access. Never returns the matched
    substring — only the category name — so a caller cannot echo the secret.
    """
    for category, pattern in _CONTENT_PATTERNS:
        if pattern.search(data):
            return category
    return is_sensitive(data)

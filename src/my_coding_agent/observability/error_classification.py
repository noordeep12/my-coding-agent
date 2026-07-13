"""Shared error-class normalization for tool-call failures.

The single source of the classification rule so a recorded `tool_call`
event's `error_class` and the anomaly detector's failure signature
(`pipeline/nodes/anomaly_detect.py`) always agree — see design D2 of
`tool-call-outcome-fields`.
"""

from __future__ import annotations

import re

# Last Python-exception-style token in an error string, e.g. the
# ``json.decoder.JSONDecodeError`` in a bash traceback. Matches a dotted
# identifier ending in ``Error`` or ``Exception``.
_EXC_TOKEN_RE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception)\b")

# Digits stripped from the fallback bucket so wording that differs only by a
# number (line numbers, counts) still buckets together.
_DIGITS_RE = re.compile(r"\d+")

_FALLBACK_MAX_LEN = 80


def classify_error(error_text: str) -> str:
    """Return the normalized error class for ``error_text``.

    The last ``…Error``/``…Exception``-style token found in ``error_text``
    when one is present (e.g. ``json.decoder.JSONDecodeError`` inside a bash
    traceback); otherwise the text's first line, digits stripped and
    truncated, as a crude bucket.
    """
    matches: list[str] = _EXC_TOKEN_RE.findall(error_text)
    if matches:
        return matches[-1]
    first_line = error_text.splitlines()[0] if error_text else ""
    normalized: str = _DIGITS_RE.sub("", first_line).strip()
    return normalized[:_FALLBACK_MAX_LEN]

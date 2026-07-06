"""Tool-output post-processing: truncation and artifact preview building.

Pure helpers the executor composes after dispatch — no LLM client, no
execution state.
"""

import json

from ...utils import get_logger
from .lang import looks_like_json
from .schema import (
    EXTRACTION_OUTPUT_TOKEN_BUDGET,
    MAX_TOOL_OUTPUT_CHARS,
    PREVIEW_MAX_CHARS,
)

logger = get_logger(__name__)

# Bounds the confirmation json.loads() in looks_like_json so a pathological
# multi-megabyte single-line artifact can't stall the (otherwise sub-ms) preview
# build. Comfortably above real-world artifact sizes seen in practice.
_JSON_SNIFF_MAX_CHARS = 2_000_000


def validate_tool_output(
    result: str,
    func_name: str,
    session_log_path: str | None,
    is_summary: bool = False,
) -> str:
    """Replace empty output and truncate oversized output to the limit."""
    if not result.strip():
        return "(tool returned empty output)"
    if not is_summary and len(result) > MAX_TOOL_OUTPUT_CHARS:
        log_hint = (
            f" Use read_file(file_path='{session_log_path}') "
            "to inspect the full output."
            if session_log_path
            else ""
        )
        logger.warning(
            "tool output truncated: %s returned %d chars (limit %d)",
            func_name,
            len(result),
            MAX_TOOL_OUTPUT_CHARS,
        )
        result = (
            result[:MAX_TOOL_OUTPUT_CHARS]
            + f"\n[output truncated at {MAX_TOOL_OUTPUT_CHARS} chars —"
            + f" full output is in the session log.{log_hint}]"
        )
    if func_name == "bash" and not is_summary:
        try:
            json.loads(result.split("\n[output truncated")[0])
        except json.JSONDecodeError:
            logger.warning("bash tool returned non-JSON output")
    return result


_DISCLOSURE = (
    "Note: read_tool_artifact(query=...) is a separate LLM call, bounded by "
    f"an output cap of ~{EXTRACTION_OUTPUT_TOKEN_BUDGET} tokens, and may scan "
    "only part of a large artifact — weigh that before choosing it over a "
    "mechanical read."
)


def _skim_guidance(
    full_output_path: str | None, preview: dict[str, int], is_json: bool
) -> str:
    """Build the inline guidance that steers the model to bounded access paths.

    States the artifact's shape (total bytes/lines) so the model can judge which
    access path is viable. For structured (JSON) content with an on-disk file,
    leads with the deterministic mechanical path (``jq``/``python`` over that
    file) ahead of the query-scoped and byte-range ``read_tool_artifact`` modes
    — including for single-line JSON, where raw byte ranges cannot address a
    value inside nested structure. Non-structured content keeps the prior
    behavior: pattern/slice bash text tools for multi-line, byte-range-only for
    single-line (or near it). Never suggests a whole-file read. Every path ends
    with a disclosure of query mode's true cost (a capped, possibly-partial LLM
    call).
    """
    total_lines = preview["total_lines"]
    total_bytes = preview["total_bytes"]
    shape = f"total {total_bytes} bytes, {total_lines} line(s)"
    query_path = (
        'read_tool_artifact(tool_call_id=..., query="what you need") for a '
        "bounded, query-scoped extract"
    )
    range_path = (
        "read_tool_artifact(tool_call_id=..., start=<offset>, length=<n>) "
        "for an exact, verbatim byte-range slice"
    )

    if is_json and full_output_path:
        p = full_output_path
        mechanical_path = (
            f"This output is JSON. Read it with a deterministic mechanical "
            f"path over the full file at {p} as your first move — e.g. "
            f"jq '<filter>' {p}, or python -c \"import json; "
            f"d=json.load(open('{p}')); ...\" — never read the whole file "
            f"at once. Only fall back to {query_path}, or {range_path}, if "
            "the mechanical path doesn't fit."
        )
        return (
            f"[Preview: {shape}, showing {preview['shown_bytes']}/{total_bytes} "
            f"bytes. Do NOT assume the excerpt is everything. {mechanical_path} "
            f"{_DISCLOSURE}]"
        )

    modes = f"Use {query_path}, or {range_path}."
    if is_json:
        # JSON detected but no on-disk file to target mechanically (D4): fall
        # back to the bounded read_tool_artifact paths.
        fallback = (
            "This output is JSON, but no on-disk artifact file is available "
            "for a mechanical read — "
        )
        return (
            f"[Preview: {shape}, showing {preview['shown_bytes']}/{total_bytes} "
            f"bytes. Do NOT assume the excerpt is everything. {fallback}"
            f"{modes} {_DISCLOSURE}]"
        )

    if total_lines <= 1:
        line_warning = (
            "This output is a single line (or near it), so line-oriented tools "
            "(head/tail, wc -l, sed line ranges) cannot bound it — "
            f"{range_path} is the way to read it in pieces."
        )
        return (
            f"[Preview: {shape}, showing {preview['shown_bytes']}/{total_bytes} "
            f"bytes. Do NOT assume the excerpt is everything. {line_warning} "
            f"{modes} {_DISCLOSURE}]"
        )

    counts = (
        f"showing {preview['shown_lines']}/{total_lines} lines, "
        f"{preview['shown_bytes']}/{total_bytes} bytes"
    )
    if full_output_path:
        p = full_output_path
        pattern_tools = (
            f"Pattern/slice bash text tools over the full file at {p} remain "
            f"available if you already know what you're looking for (grep/rg "
            f"'<pattern>' {p}; sed -n '<start>,<end>p' {p}; awk; jq) — never "
            "read the whole file at once."
        )
        return (
            f"[Preview: {shape}, {counts}. Do NOT assume the excerpt is "
            f"everything — {modes} {pattern_tools} {_DISCLOSURE}]"
        )
    return (
        f"[Preview: {shape}, {counts}. Do NOT assume the excerpt is "
        f"everything — {modes} {_DISCLOSURE}]"
    )


def build_stream_preview(
    text: str, full_output_path: str | None
) -> tuple[str, dict[str, int | str | None]]:
    """Build the agent-facing field value and the ``preview`` descriptor for a stream.

    Returns ``(value, preview)`` where ``value`` is a token-bounded excerpt of the
    stream followed by inline skim guidance, and ``preview`` carries the shown/total
    line and byte counts plus the full-output file path. The full stream is never
    returned — only the bounded excerpt. Used for stdout (→ ``output``) and stderr
    (→ ``error``) alike.
    """
    total_bytes = len(text)
    total_lines = text.count("\n") + 1 if text else 0

    excerpt = text[:PREVIEW_MAX_CHARS]
    if len(text) > PREVIEW_MAX_CHARS:
        cut = excerpt.rfind("\n")
        if cut > 0:
            excerpt = excerpt[:cut]  # trim to a whole line for readability
    shown_bytes = len(excerpt)
    shown_lines = excerpt.count("\n") + 1 if excerpt else 0

    counts = {
        "shown_lines": shown_lines,
        "total_lines": total_lines,
        "shown_bytes": shown_bytes,
        "total_bytes": total_bytes,
    }
    is_json = looks_like_json(text, max_chars=_JSON_SNIFF_MAX_CHARS)
    guidance = _skim_guidance(full_output_path, counts, is_json)
    output = f"{excerpt}\n\n{guidance}" if excerpt else guidance
    preview: dict[str, int | str | None] = {
        **counts,
        "full_output_path": full_output_path,
    }
    return output, preview

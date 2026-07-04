"""Tool-output post-processing: truncation and artifact preview building.

Pure helpers the executor composes after dispatch — no LLM client, no
execution state.
"""

import json

from ...utils import get_logger
from .schema import MAX_TOOL_OUTPUT_CHARS, PREVIEW_MAX_CHARS

logger = get_logger(__name__)


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


def _skim_guidance(full_output_path: str | None, preview: dict[str, int]) -> str:
    """Build the inline guidance that steers the model to query-scoped access.

    Primary path: ``read_tool_artifact(tool_call_id=..., query=...)``, which
    returns a bounded extract relevant to the query — never the whole output.
    Bash text tools over the on-disk file remain a secondary path for callers
    who already know the shape of what they need (a line number, an exact
    pattern).
    """
    counts = (
        f"showing {preview['shown_lines']}/{preview['total_lines']} lines, "
        f"{preview['shown_bytes']}/{preview['total_bytes']} bytes"
    )
    primary = (
        "Do NOT assume the excerpt is everything — to pull out a specific "
        'detail, call read_tool_artifact(tool_call_id=..., query="what you '
        'need") — it returns a bounded extract, never the full output.'
    )
    if full_output_path:
        p = full_output_path
        return (
            f"[Preview: {counts}. {primary} Full output also on disk at {p} — "
            f"skim it with bash text tools (grep/rg '<pattern>' {p}; "
            f"sed -n '<start>,<end>p' {p}; awk; jq; head/tail; wc -l {p}) if you "
            "already know what you're looking for.]"
        )
    return f"[Preview: {counts}. {primary}]"


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
    guidance = _skim_guidance(full_output_path, counts)
    output = f"{excerpt}\n\n{guidance}" if excerpt else guidance
    preview: dict[str, int | str | None] = {
        **counts,
        "full_output_path": full_output_path,
    }
    return output, preview

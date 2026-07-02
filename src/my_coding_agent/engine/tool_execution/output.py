"""Tool-output post-processing: summary extraction, truncation, artifact summary.

Helpers the executor composes after dispatch. ``_extract_summary`` and
``validate_tool_output`` are pure; ``summarize_artifact`` takes the LLM client
explicitly (an injected dependency) so this module owns no execution state.
"""

import json
import re
from typing import TYPE_CHECKING

from ...utils import get_logger
from ...utils.parsing import extract_message
from ..tool_registry import ARTIFACT_THRESHOLD

if TYPE_CHECKING:
    from ..llm import LLM

logger = get_logger(__name__)

# Single source of truth lives in tools.ARTIFACT_THRESHOLD: the artifact-separation
# boundary and this truncation boundary are the same concept (large tool output).
MAX_TOOL_OUTPUT_CHARS = ARTIFACT_THRESHOLD

# Preview budget for an offloaded artifact: only a bounded excerpt goes into the
# tool result `output`; the full content stays on disk. Kept well under
# ARTIFACT_THRESHOLD so the preview never itself approaches the offload boundary.
PREVIEW_TOKEN_BUDGET = 500  # approx. tokens shown in the preview excerpt
_CHARS_PER_TOKEN = 4  # rough chars/token estimate used only for budgeting
PREVIEW_MAX_CHARS = PREVIEW_TOKEN_BUDGET * _CHARS_PER_TOKEN

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _extract_summary(content: str) -> str:
    """Return only the summary, dropping any model thinking/preamble.

    Prefers the explicit ``<summary>...</summary>`` block the prompt asks for;
    otherwise strips ``<think>...</think>`` reasoning blocks and returns the rest.
    """
    match = _SUMMARY_RE.search(content)
    if match:
        return match.group(1).strip()
    return _THINK_RE.sub("", content).strip()


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
    """Build the inline guidance that steers the model to skim, not load whole."""
    counts = (
        f"showing {preview['shown_lines']}/{preview['total_lines']} lines, "
        f"{preview['shown_bytes']}/{preview['total_bytes']} bytes"
    )
    if full_output_path:
        p = full_output_path
        return (
            f"[Preview: {counts}. Full output on disk at {p}. Do NOT read the whole "
            f"file — skim it with bash text tools: grep/rg '<pattern>' {p}; "
            f"sed -n '<start>,<end>p' {p}; awk; jq (JSON); head/tail; wc -l {p}.]"
        )
    return (
        f"[Preview: {counts}. Do NOT load the whole output — inspect only what you "
        "need via read_tool_artifact(tool_call_id=...).]"
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
    guidance = _skim_guidance(full_output_path, counts)
    output = f"{excerpt}\n\n{guidance}" if excerpt else guidance
    preview: dict[str, int | str | None] = {
        **counts,
        "full_output_path": full_output_path,
    }
    return output, preview


def summarize_artifact(
    client: "LLM",
    artifact: dict,
    func_name: str,
    tool_call_id: str,
) -> str:
    """Summarize an offloaded artifact for the model, pointing at the full copy."""
    logger.tool(
        "%s → %s: artifact %s chars (summarizing for model)",
        tool_call_id,
        func_name,
        len(json.dumps(artifact)),
    )
    prompt = (
        "/no_think\n"
        f"Summarize the following `{func_name}` tool output concisely "
        "for an AI coding agent. "
        "Include: exit status, key findings, any errors, and what the "
        "agent needs to know to continue its task. "
        "Be factual and brief — 3 to 8 sentences max.\n"
        "Output ONLY the summary itself — no reasoning, analysis, planning, "
        "or preamble. Wrap the summary in <summary>...</summary> tags.\n\n"
        f"Output:\n{json.dumps(artifact, indent=2)[:12_000]}"
    )
    try:
        resp = client.chat_completion(
            [{"role": "user", "content": prompt}],
            tools=[],
            kind="tool_output_summarizer",
            max_tokens=512,
        )
        summary = _extract_summary(extract_message(resp).get("content") or "")
    except Exception as exc:
        logger.warning("artifact summarization failed: %s", exc)
        if "content" in artifact:
            summary = json.dumps(
                {
                    "file_path": artifact.get("file_path"),
                    "size": artifact.get("size"),
                }
            )
        else:
            summary = json.dumps(
                {
                    "exit_code": artifact.get("exit_code"),
                    "ok": artifact.get("ok"),
                    "stdout_chars": len(artifact.get("stdout", "")),
                    "stderr_chars": len(artifact.get("stderr", "")),
                }
            )
    return summary.strip() + (
        f"\n\n[Full output stored as artifact — use "
        f'read_tool_artifact(tool_call_id="{tool_call_id}") ONLY if the '
        "summary above is insufficient to proceed. "
        "Avoid calling it unless strictly necessary.]"
    )

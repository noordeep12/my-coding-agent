"""Tool-output post-processing: summary extraction, truncation, artifact summary.

Helpers the executor composes after dispatch. ``_extract_summary`` and
``validate_tool_output`` are pure; ``summarize_artifact`` takes the LLM client
explicitly (an injected dependency) so this module owns no execution state.
"""

import json
import re
from typing import TYPE_CHECKING

from ..logger import get_logger
from ..tool_registry import ARTIFACT_THRESHOLD
from ..utils import extract_message

if TYPE_CHECKING:
    from ..llm import LLM

logger = get_logger(__name__)

# Single source of truth lives in tools.ARTIFACT_THRESHOLD: the artifact-separation
# boundary and this truncation boundary are the same concept (large tool output).
MAX_TOOL_OUTPUT_CHARS = ARTIFACT_THRESHOLD

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


def describe_artifact(artifact: dict, tool_call_id: str) -> str:
    """Deterministically describe an offloaded artifact (no LLM), with a pointer.

    Reports just enough for the agent to decide whether to fetch the full output
    via ``read_tool_artifact`` — file metadata for file reads, exit status and
    byte counts for command output.
    """
    if "content" in artifact:
        summary = json.dumps(
            {"file_path": artifact.get("file_path"), "size": artifact.get("size")}
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
    return summary + (
        f"\n\n[Full output stored as artifact — use read_tool_artifact("
        f'tool_call_id="{tool_call_id}") to inspect it.]'
    )


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

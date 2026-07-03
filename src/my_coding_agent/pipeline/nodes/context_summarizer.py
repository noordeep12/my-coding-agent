"""ContextSummarizerNode — tool-free full-conversation summarization on trigger."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ...engine.llm.schema import CALL_KIND_HANDOFF, CALL_KIND_REPORT
from ...utils import get_logger
from ...utils.parsing import extract_message
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)

REPORT_PROMPT = (
    "Your task is complete. Write your final report now, as plain text, for the "
    "agent that delegated this task to you. Do NOT call any tools and do NOT "
    "continue working — respond with the report text only.\n\n"
    "Include:\n"
    "1. **Task** — what you were asked to do\n"
    "2. **Findings** — the key results, answers, and evidence you gathered "
    "(be specific: file paths, names, values, quotes)\n"
    "3. **Conclusion** — a direct answer to the task\n\n"
    "Be concise and self-contained: the delegating agent sees only this report, "
    "not your conversation. Do not describe remaining work or a continuation — "
    "this is your final output."
)

HANDOFF_PROMPT = (
    "CONTEXT RESET REQUIRED: your context window is nearly full. "
    "Before the reset, write a structured handoff so the continuation agent "
    "can pick up exactly where you left off.\n\n"
    "Include ALL of the following:\n"
    "1. **Original task** — what was asked\n"
    "2. **Progress** — what you have completed so far (be specific)\n"
    "3. **Files created/modified** — list each file and what it contains\n"
    "4. **Key decisions** — important choices made and why\n"
    "5. **Remaining work** — exactly what still needs to be done, in order\n"
    "6. **Critical context** — any state the next agent must know "
    "to continue correctly\n\n"
    "Be exhaustive. This will be the ONLY context the continuation agent starts with."
)

# Summarization kind → prompt. CALL_KIND_REPORT synthesizes a hand-back for the
# delegating parent; CALL_KIND_HANDOFF seeds a continuation agent after a
# context reset.
_PROMPTS: dict[str, str] = {
    CALL_KIND_REPORT: REPORT_PROMPT,
    CALL_KIND_HANDOFF: HANDOFF_PROMPT,
}


def summarize_conversation(
    llm: Any, messages: list[dict[str, Any]], prompt: str, kind: str
) -> str:
    """Summarize *messages* via one tool-free LLM call and return the text.

    Appends *prompt* as a user turn and issues a single chat completion with no
    tools, tagged *kind*. Shared by ``ContextSummarizerNode`` (in-pipeline) and
    ``AgentNode.generate_report`` (out-of-pipeline fallback) so every
    summarization goes through one path.

    Args:
        llm: The run's LLM client.
        messages: The conversation to summarize (not mutated).
        prompt: Instruction appended as the final user message.
        kind: Call-kind tag for token accounting and the trace.

    Returns:
        The assistant's summary text (empty string if none was produced).
    """
    summary_messages = messages + [{"role": "user", "content": prompt}]
    resp = llm.chat_completion(summary_messages, tools=[], kind=kind)
    message = extract_message(resp)
    # Reasoning models (e.g. Qwen3-thinking) often end the summary turn with a
    # tool call or bare thinking, leaving ``content`` empty while the actual
    # summary lives in ``reasoning_content``. Fall back to it so the summary is
    # never lost to an empty ``content`` field.
    content = message.get("content") or ""
    if not content.strip():
        content = message.get("reasoning_content") or ""
    return content


class ContextSummarizerNode(BaseNode):
    """Summarize the full conversation when another node triggers it.

    Not a member of the ordered step list: the node that decides the condition
    (``FinalizeStepNode`` for the step-ceiling cutoff, ``ContextGuardNode`` for
    the context-reset handoff and the context-limit stop) holds an instance and
    calls ``run(ctx)`` when its condition fires. The summary is written to
    ``ctx.handback_report`` (kind ``report``) or ``ctx.handoff_content`` (kind
    ``handoff``), and one ``summarizer`` event is recorded carrying kind, step,
    the triggering node's name, latency, and token usage — the trace link the
    viewer uses to nest this node under its trigger.
    """

    name = "context_summarizer"

    def __init__(self, kind: str, triggered_by: str) -> None:
        """Bind the summarization *kind* and the *triggered_by* node name."""
        self._kind = kind
        self._triggered_by = triggered_by

    def run(self, ctx: RunContext) -> None:
        _logger.info(
            "Summarizing conversation (kind=%s, triggered by %s)...",
            self._kind,
            self._triggered_by,
        )
        started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        t0 = time.monotonic()
        content = summarize_conversation(
            ctx.llm, ctx.messages, _PROMPTS[self._kind], self._kind
        )
        latency = time.monotonic() - t0
        if self._kind == CALL_KIND_REPORT:
            ctx.handback_report = content
        else:
            ctx.handoff_content = content
        last_call = ctx.llm.llm_calls[-1] if ctx.llm.llm_calls else {}
        ctx.recorder.record_summarizer(
            kind=self._kind,
            step=ctx.step_num,
            triggered_by=self._triggered_by,
            latency_s=latency,
            prompt_tokens=last_call.get("prompt", 0),
            completion_tokens=last_call.get("completion", 0),
            total_tokens=last_call.get("total", 0),
            started_at=started_at,
        )

"""AgentNode — run the full agentic pipeline as a composable node."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..observability import Recorder, current_session_id
from ..observability.recorder import current_recorder
from ..pipeline.context import RunContext
from ..pipeline.node import BaseNode
from ..pipeline.nodes.handoff import ContextHandoff
from ..utils import (
    attach_session_log,
    detach_session_log,
    get_logger,
    print_banner,
    print_run_summary,
)
from ..utils.parsing import extract_message
from .llm import LLM, OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL

_HANDOFF_PROMPT = (
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

_REPORT_PROMPT = (
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


class AgentNode(BaseNode):
    """Run the agentic pipeline as a composable node or stand-alone via execute().

    Owns session bookkeeping (banner, session log, summary, session_data.json).
    When embedded in a larger pipeline, call ``run(ctx)``; it drives the inner
    step loop and sets ``ctx.signal = "STOP"`` when done.  For stand-alone use,
    call ``execute(max_steps)`` which creates the ``RunContext`` internally.
    """

    name = "agent"

    def __init__(
        self,
        api_url: str = OMLX_API_URL,
        api_key: str = OMLX_API_KEY,
        model: str = OMLX_MODEL,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        label: str = "Agent",
        context_reset_threshold: float = 0.75,
    ) -> None:
        """Initialize the agent, open a session log, and build the LLM client.

        No network I/O is performed; the LLM context-window probe is deferred.
        """
        self.session_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now().isoformat(timespec="seconds")
        _session_dir = Path(".my_coding_agent") / self.session_id
        self.recorder = Recorder(
            self.session_id, _session_dir, parent_session_id=current_session_id.get()
        )
        self.llm = LLM(api_url, api_key, model)
        self.llm._recorder = self.recorder
        self.tool_artifacts: dict = {}
        self.label = label
        self.messages = messages or []
        self.tools = tools or []
        self.context_reset_threshold = context_reset_threshold
        self.logger = get_logger(self.__class__.__name__)
        _log_path = _session_dir / "stderr.log"
        self.llm._session_log_path = str(_log_path)
        self._session_log_handler = attach_session_log(_log_path)
        self.step_num = 0
        self.stop_reason = "max_steps"
        self.tool_records: list = []
        self.handoff_records: list = []
        self.elapsed_seconds: float = 0.0
        self.last_prompt_tokens: int = 0
        self.logger.info(
            "%s initialized with %d messages and %d tools",
            label,
            len(self.messages),
            len(self.tools),
        )

    def run(self, ctx: RunContext) -> None:
        """Run the full agentic loop and write results back to ctx."""
        messages = self.execute(max_steps=ctx.max_steps)
        ctx.messages = messages
        ctx.step_num = self.step_num
        ctx.stop_reason = self.stop_reason
        ctx.tool_records = self.tool_records
        ctx.handoff_records = self.handoff_records
        ctx.tool_artifacts = self.tool_artifacts
        ctx.last_prompt_tokens = self.last_prompt_tokens
        ctx.signal = "STOP"

    def execute(self, max_steps: int = 5) -> list[dict[str, Any]]:
        """Drive the agentic pipeline and return the final message list."""
        from ..pipeline import build_default_pipeline

        self.step_num = 0
        self.stop_reason = "max_steps"
        self.tool_records = []
        self.handoff_records = []
        self.llm.llm_calls = []
        self.last_prompt_tokens = 0

        t_start = time.monotonic()
        self.recorder.start(self.label, self.llm.model, self.llm.context_window)
        _ctx_token = current_session_id.set(self.session_id)
        _rec_token = current_recorder.set(self.recorder)

        print_banner(
            label=self.label,
            model=self.llm.model,
            tools=self.tools,
            context_window=self.llm.context_window,
            n_messages=len(self.messages),
            context_reset_threshold=self.context_reset_threshold,
            session_id=self.session_id,
        )
        self.logger.info("Agent run started with max_steps: %d", max_steps)

        ctx = RunContext(
            session_id=self.session_id,
            label=self.label,
            max_steps=max_steps,
            context_reset_threshold=self.context_reset_threshold,
            all_tools=self.tools,
            llm=self.llm,
            recorder=self.recorder,
            messages=self.messages,
            last_prompt_tokens=self.last_prompt_tokens,
        )

        def _spawn_fn() -> list[dict[str, Any]]:
            return self._handle_context_reset(ctx, max_steps, t_start)

        pipeline = build_default_pipeline(spawn_fn=_spawn_fn)

        result: list[dict[str, Any]] = []
        try:
            result = pipeline.execute(ctx)
        except KeyboardInterrupt:
            ctx.stop_reason = "aborted"
            self.logger.warning("Agent run aborted by user (KeyboardInterrupt)")
        finally:
            self._sync_from_ctx(ctx, t_start)
            out = Path(".my_coding_agent") / self.session_id / "session_data.json"
            if not out.exists():
                self._save_session_data(max_steps)
                self._print_summary(max_steps)
                detach_session_log(self._session_log_handler)
            self.recorder.finish(
                self.stop_reason, self.step_num, round(self.elapsed_seconds, 3)
            )
            current_session_id.reset(_ctx_token)
            current_recorder.reset(_rec_token)

        return result if result else ctx.messages

    def _sync_from_ctx(self, ctx: RunContext, t_start: float) -> None:
        self.step_num = ctx.step_num
        self.stop_reason = ctx.stop_reason
        self.tool_records = ctx.tool_records
        self.handoff_records = ctx.handoff_records
        self.tool_artifacts = ctx.tool_artifacts
        self.last_prompt_tokens = ctx.last_prompt_tokens
        self.elapsed_seconds = time.monotonic() - t_start

    def _summarize_conversation(self, prompt: str, kind: str) -> str:
        """Summarize the running conversation via one tool-free LLM call.

        Appends *prompt* as a user turn to the current messages and issues a
        single chat completion with no tools, tagged *kind*, returning the
        assistant text. Shared by the context-reset handoff and the subagent
        end-of-turn report so both go through one summarization path.

        Args:
            prompt: Instruction appended as the final user message.
            kind: Call-kind tag for token accounting and the trace.

        Returns:
            The assistant's summary text (empty string if none was produced).
        """
        summary_messages = self.messages + [{"role": "user", "content": prompt}]
        resp = self.llm.chat_completion(summary_messages, tools=[], kind=kind)
        message = extract_message(resp)
        # Reasoning models (e.g. Qwen3-thinking) often end the summary turn with a
        # tool call or bare thinking, leaving ``content`` empty while the actual
        # summary lives in ``reasoning_content``. Fall back to it so the summary is
        # never lost to an empty ``content`` field.
        content = message.get("content") or ""
        if not content.strip():
            content = message.get("reasoning_content") or ""
        return content

    def generate_report(self) -> str:
        """Summarize the whole run as a final report and record it as a node.

        Issues one tool-free LLM call over the full conversation with a
        report-specific prompt, records the result as a distinct report node,
        and returns it. Reused by ``delegate`` so the main agent receives a
        complete synthesized report rather than a scraped last message. Falls
        back to a placeholder when the model returns nothing, so the caller
        never receives an empty report.

        Returns:
            The final report text (never empty).
        """
        self.logger.info("Generating subagent report summary...")
        content = self._summarize_conversation(_REPORT_PROMPT, "report")
        if not content.strip():
            content = "(subagent produced no report)"
        self.recorder.record_report(content)
        return content

    def _generate_handoff(self, step_num: int, prompt_tokens: int) -> ContextHandoff:
        """Ask the LLM to summarize current state, persist the handoff, return it."""
        self.logger.info("Generating context handoff summary...")
        content = self._summarize_conversation(_HANDOFF_PROMPT, "handoff")
        handoff = ContextHandoff(
            agent_label=self.label,
            step_num=step_num,
            prompt_tokens=prompt_tokens,
            context_window=self.llm.context_window,
            content=content,
        )
        path = handoff.save()
        self.logger.info("Handoff saved → %s", path)
        return handoff

    def _spawn_continuation(
        self, handoff: ContextHandoff, max_steps: int
    ) -> list[dict[str, Any]]:
        system_messages = [m for m in self.messages if m.get("role") == "system"]
        continuation = AgentNode(
            api_url=self.llm.api_url,
            api_key=self.llm.api_key,
            model=self.llm.model,
            messages=system_messages + [handoff.to_user_message()],
            tools=self.tools,
            label=f"{self.label} (cont.)",
            context_reset_threshold=self.context_reset_threshold,
        )
        remaining_steps = max_steps - self.step_num
        return continuation.execute(max_steps=max(remaining_steps, 1))

    def _print_summary(self, max_steps: int) -> None:
        last_message = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_message = msg["content"]
                break
        calls = self.llm.llm_calls
        print_run_summary(
            steps=self.step_num,
            max_steps=max_steps,
            stop_reason=self.stop_reason,
            prompt_tokens=sum(c["prompt"] for c in calls),
            completion_tokens=sum(c["completion"] for c in calls),
            total_tokens=sum(c["total"] for c in calls),
            last_prompt_tokens=self.last_prompt_tokens,
            context_window=self.llm.context_window,
            elapsed_seconds=self.elapsed_seconds,
            tool_records=self.tool_records,
            handoff_records=self.handoff_records,
            agent_name=self.label,
            last_message=last_message,
            llm_calls=calls,
            model=self.llm.model,
            session_id=self.session_id,
            started_at=self.started_at,
            tools=self.tools,
        )

    def _save_session_data(self, max_steps: int) -> None:
        last_message = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_message = msg["content"]
                break
        data = {
            "session_id": self.session_id,
            "agent_label": self.label,
            "model": self.llm.model,
            "started_at": self.started_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "steps": self.step_num,
            "max_steps": max_steps,
            "stop_reason": self.stop_reason,
            "total_usage": {
                "prompt_tokens": sum(c["prompt"] for c in self.llm.llm_calls),
                "completion_tokens": sum(c["completion"] for c in self.llm.llm_calls),
                "total_tokens": sum(c["total"] for c in self.llm.llm_calls),
            },
            "context_window": self.llm.context_window,
            "context_reset_threshold": self.context_reset_threshold,
            "tool_records": self.tool_records,
            "handoff_records": self.handoff_records,
            "llm_calls": self.llm.llm_calls,
            "last_message": last_message,
        }
        out = Path(".my_coding_agent") / self.session_id / "session_data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        self.logger.info("Session data saved → %s", out)
        if self.tool_artifacts:
            artifacts_out = out.parent / "tool_artifacts.json"
            artifacts_out.write_text(json.dumps(self.tool_artifacts, indent=2))
            self.logger.info("Tool artifacts saved → %s", artifacts_out)

    def _handle_context_reset(
        self,
        ctx: RunContext,
        max_steps: int,
        t_start: float,
    ) -> list[dict[str, Any]]:
        """Generate a handoff, finalize this run, spawn and return the continuation."""
        ctx_tokens = ctx.last_prompt_tokens or len(json.dumps(ctx.messages)) // 2
        ctx_pct = ctx_tokens / ctx.llm.context_window if ctx.llm.context_window else 0.0

        handoff = self._generate_handoff(ctx.step_num, ctx_tokens)
        ctx.handoff_records.append(
            {
                "step": ctx.step_num,
                "ctx_tokens": ctx_tokens,
                "ctx_pct": ctx_pct * 100,
                "threshold": ctx.context_reset_threshold * 100,
                "path": handoff.path,
                "reason": (
                    f"prompt_tokens {ctx_tokens:,} >= "
                    f"{ctx.context_reset_threshold * 100:.0f}% "
                    f"of {ctx.llm.context_window:,}"
                ),
            }
        )
        self.recorder.record_handoff(
            ctx.step_num,
            ctx_tokens,
            ctx_pct * 100,
            handoff.content,
            handoff.path or "",
        )
        ctx.stop_reason = "context_reset"
        self._sync_from_ctx(ctx, t_start)
        self._save_session_data(max_steps)
        self._print_summary(max_steps)
        detach_session_log(self._session_log_handler)
        return self._spawn_continuation(handoff, max_steps)

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
from ..pipeline.handoff import handoff_to_user_message, save_handoff
from ..pipeline.node import BaseNode
from ..pipeline.nodes.context_summarizer import (
    HANDOFF_PROMPT,
    REPORT_PROMPT,
    summarize_conversation,
)
from ..pipeline.schema import ContextHandoff
from ..utils import (
    attach_session_log,
    detach_session_log,
    get_logger,
    print_banner,
    print_run_summary,
)
from .llm import LLM, OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL
from .llm.schema import CALL_KIND_HANDOFF, CALL_KIND_REPORT

# Default step budget shared by the main agent (CLI), the ``execute`` default,
# and delegated subagents, so all three run with the same ceiling.
DEFAULT_MAX_STEPS = 50


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
        needs_handback: bool = False,
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
        self.needs_handback = needs_handback
        self.handback_report: str | None = None
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

    def execute(self, max_steps: int = DEFAULT_MAX_STEPS) -> list[dict[str, Any]]:
        """Drive the agentic pipeline and return the final message list."""
        from ..pipeline import build_default_pipeline

        self.step_num = 0
        self.stop_reason = "max_steps"
        self.tool_records = []
        self.handoff_records = []
        self.llm.llm_calls = []
        self.last_prompt_tokens = 0
        self.handback_report = None

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
            needs_handback=self.needs_handback,
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
        self.handback_report = ctx.handback_report
        self.elapsed_seconds = time.monotonic() - t_start

    def final_assistant_text(self) -> str:
        """Return the last assistant message's text, or empty string.

        Applies the same fallback as summarization: use ``content``, else
        ``reasoning_content`` when a reasoning model left ``content`` empty.
        Used by ``delegate`` to hand back a cleanly finished subagent's final
        turn verbatim instead of paying a full-conversation summarization.
        """
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content") or ""
                if not content.strip():
                    content = msg.get("reasoning_content") or ""
                return content
        return ""

    def generate_report(self) -> str:
        """Summarize the whole run as a final report and record it as a node.

        Issues one tool-free LLM call over the full conversation with a
        report-specific prompt, records the result as a distinct report node,
        and returns it. Out-of-pipeline fallback for ``delegate`` when the
        pipeline produced no hand-back (e.g. an aborted run). Falls back to a
        placeholder when the model returns nothing, so the caller never
        receives an empty report.

        Returns:
            The final report text (never empty).
        """
        self.logger.info("Generating subagent report summary...")
        content = summarize_conversation(
            self.llm, self.messages, REPORT_PROMPT, CALL_KIND_REPORT
        )
        if not content.strip():
            content = "(subagent produced no report)"
        self.recorder.record_report(content)
        return content

    def _generate_handoff(
        self, step_num: int, prompt_tokens: int, content: str = ""
    ) -> ContextHandoff:
        """Persist the handoff summary (generating it only if not provided).

        *content* normally arrives pre-produced by ``ContextSummarizerNode``
        (triggered by ``ContextGuardNode``); the summarization here is a
        fallback for callers that reach this path without the pipeline.
        """
        if not content.strip():
            self.logger.info("Generating context handoff summary...")
            content = summarize_conversation(
                self.llm, self.messages, HANDOFF_PROMPT, CALL_KIND_HANDOFF
            )
        handoff = ContextHandoff(
            agent_label=self.label,
            step_num=step_num,
            prompt_tokens=prompt_tokens,
            context_window=self.llm.context_window,
            content=content,
        )
        path = save_handoff(handoff)
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
            messages=system_messages + [handoff_to_user_message(handoff)],
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

        handoff = self._generate_handoff(
            ctx.step_num, ctx_tokens, ctx.handoff_content or ""
        )
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

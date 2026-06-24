"""Agentic loop built on top of the LLM client.

Defines ``Agent``, which holds an ``LLM`` client (composition) and runs the
agentic pipeline via ``pipeline.execute(ctx)``.  The pipeline is a DAG of
named nodes (context preflight, tool routing, LLM call, tool dispatch, token
tracking, finish check); ``run`` constructs a ``RunContext``, builds the
pipeline, and delegates execution to it.  Session bookkeeping (banner, session
log, summary, session_data.json) stays on ``Agent``.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .handoff import ContextHandoff
from .llm import LLM, OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL
from .logger import (
    attach_session_log,
    detach_session_log,
    get_logger,
    print_banner,
    print_run_summary,
)
from .observability import Recorder, current_session_id
from .observability.recorder import current_recorder
from .pipeline import RunContext, build_default_pipeline
from .utils import extract_message

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


class Agent:
    """Run the agentic pipeline over a held ``LLM`` client.

    Hold an ``LLM`` client (``self.llm``) via composition and drive the pipeline
    in ``run(max_steps)``.  The pipeline is built from six named nodes; this class
    owns session bookkeeping (banner, session log, summary, session_data.json) and
    the ``_spawn_continuation`` callable passed into ``ContextPreflightNode``.
    """

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
        """Initialize the agent and open a session log (no network I/O).

        Build the held ``LLM`` client (whose context-window probe is deferred),
        assign a fresh session id, redirect stderr into per-session log files via
        ``attach_session_log``, and initialize run statistics. Construction
        performs no network call; the startup banner is printed at the start of
        ``run`` once the real context window is known.

        Args:
            api_url: Base URL of the OpenAI-compatible API.
            api_key: Bearer token sent on every request.
            model: Model id used for completions and context-window lookup.
            messages: Initial conversation (e.g. system prompt). None starts
                empty; the same list is appended to in place by ``add_message``.
            tools: Tool schemas available to the agent. None starts empty.
            label: Human-readable name shown in the banner and summary.
            context_reset_threshold: Prompt-token fraction of the context window
                (0-1) at which a handoff is generated and a continuation spawned.
        """
        self.session_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now().isoformat(timespec="seconds")
        _session_dir = Path(".my_coding_agent") / self.session_id
        # Observability recorder — a separate capture layer (the logger package is
        # untouched). It reads the parent session id from the contextvar so a
        # delegated subagent links back to its parent in the session tree.
        self.recorder = Recorder(
            self.session_id, _session_dir, parent_session_id=current_session_id.get()
        )
        # Hold the LLM client via composition (no longer subclass it). The
        # ToolExecutor (built per step) captures tool I/O via this recorder.
        self.llm = LLM(api_url, api_key, model)
        self.llm._recorder = self.recorder
        # A fresh ToolExecutor is built per step (per message); artifacts it
        # offloads are accumulated here for session_data.json.
        self.tool_artifacts: dict = {}
        self.label = label
        self.messages = messages or []
        self.tools = tools or []
        self.context_reset_threshold = context_reset_threshold
        self.logger = get_logger(self.__class__.__name__)
        _log_path = _session_dir / "stderr.log"
        # The executor reads the session-log path off the client to hint where
        # full truncated output lives; set it on the held client.
        self.llm._session_log_path = str(_log_path)
        self._session_log_handler = attach_session_log(_log_path)
        # run stats — reset at the start of each run()
        self.step_num = 0
        self.stop_reason = "max_steps"
        self.tool_records: list = []
        self.handoff_records: list = []  # one entry per context reset that fired
        self.elapsed_seconds: float = 0.0
        self.logger.info(
            "%s initialized with %d messages and %d tools",
            label,
            len(self.messages),
            len(self.tools),
        )

    def add_message(self, message: dict[str, Any]) -> None:
        """Append one message to the conversation in place and log its role.

        Args:
            message: An OpenAI-style message dict. A missing ``role`` is logged
                as ``UNKNOWN``; the message is appended either way.
        """
        self.messages.append(message)
        message_type = message.get("role", "unknown").upper()
        self.logger.info("%s message added to the conversation", message_type)

    def _generate_handoff(self, step_num: int, prompt_tokens: int) -> ContextHandoff:
        """Ask the LLM to summarize current state, persist handoff, return it."""
        handoff_messages = self.messages + [
            {"role": "user", "content": _HANDOFF_PROMPT}
        ]
        self.logger.info("Generating context handoff summary...")
        resp = self.llm.chat_completion(handoff_messages, tools=[], kind="handoff")
        content = extract_message(resp).get("content", "") or ""
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

    def _spawn_continuation(
        self, handoff: ContextHandoff, max_steps: int
    ) -> list[dict[str, Any]]:
        """Finish the run in a fresh agent seeded with system prompt + handoff."""
        system_messages = [m for m in self.messages if m.get("role") == "system"]
        continuation = Agent(
            api_url=self.llm.api_url,
            api_key=self.llm.api_key,
            model=self.llm.model,
            messages=system_messages + [handoff.to_user_message()],
            tools=self.tools,
            label=f"{self.label} (cont.)",
            context_reset_threshold=self.context_reset_threshold,
        )
        remaining_steps = max_steps - self.step_num
        return continuation.run(max_steps=max(remaining_steps, 1))

    def run(self, max_steps: int = 5) -> list[dict[str, Any]]:
        """Run the agentic pipeline until a stop condition, returning the messages.

        Build a ``RunContext`` and a ``Pipeline`` of six named nodes, then
        delegate execution to ``pipeline.execute(ctx)``.  Session bookkeeping
        (banner, session log, summary, session_data.json) is handled here.
        The ``KeyboardInterrupt`` path is also caught here so the finally block
        always runs.

        Args:
            max_steps: Maximum number of agent steps before stopping.

        Returns:
            The full conversation message list at the point the pipeline stopped.
        """
        # reset stats for this run
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

        # _spawn_continuation needs ctx.handoff_records / ctx.step_num at reset
        # time; it closes over `ctx` and `max_steps` from this scope.
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
            if not out.exists():  # context-reset path already saved; don't overwrite
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
        """Copy pipeline state back onto self so session-data methods see it."""
        self.step_num = ctx.step_num
        self.stop_reason = ctx.stop_reason
        self.tool_records = ctx.tool_records
        self.handoff_records = ctx.handoff_records
        self.tool_artifacts = ctx.tool_artifacts
        self.last_prompt_tokens = ctx.last_prompt_tokens
        self.elapsed_seconds = time.monotonic() - t_start

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

"""AgentNode — run the full agentic pipeline as a composable node."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..observability import Recorder, current_session_id
from ..observability.recorder import current_agent_node, current_recorder
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
from .checkpoint import Checkpoint, save_checkpoint
from .llm import LLM, OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL
from .llm.errors import LLMCallError
from .llm.schema import CALL_KIND_HANDOFF, CALL_KIND_REPORT
from .schema import REPORT_SOURCE_FALLBACK, llm_failure_stop_reason
from .tool_registry.skills import RenderedIndex, Skill, build_opening_block

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
        skills: dict[str, Skill] | None = None,
        loaded_skills: set[str] | None = None,
        place_skill_index: bool = True,
        resumed_from: str | None = None,
        resume_step: int = 0,
        resume_prompt_tokens: int = 0,
    ) -> None:
        """Initialize the agent, open a session log, and build the LLM client.

        No network I/O is performed; the LLM context-window probe is deferred.

        ``skills`` is the discovered-skill snapshot for this run (name → Skill);
        when non-empty, its index is placed into the opening user message here
        (never the system prompt, so #75 holds) and offered to the model via the
        ``use_skill`` tool. ``loaded_skills`` seeds the set of already-loaded
        skills — empty for a fresh run, or the pre-reset run's loaded names for a
        continuation, whose full bodies are re-injected into the opening message
        so knowledge survives a handoff (D6). ``place_skill_index`` is ``False``
        for a resumed run whose checkpointed opening message already carries the
        index, so ``self.skills`` still services ``use_skill`` without the index
        being placed (or its event emitted) a second time.

        ``resumed_from`` links this run to the dead session whose checkpoint
        seeded it (D5); ``resume_step``/``resume_prompt_tokens`` continue the step
        counter so the first LLM call is step N+1, not step 0. Defaults keep a
        fresh run unchanged. Prefer ``AgentNode.from_checkpoint`` over passing
        these directly.
        """
        self.session_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._session_dir = Path(".my_coding_agent") / self.session_id
        _session_dir = self._session_dir
        self.resumed_from = resumed_from
        self._resume_step = resume_step
        self._resume_prompt_tokens = resume_prompt_tokens
        # Set to the classified error when a run ends on an unrecoverable LLM
        # failure (D6), so the CLI can print a one-line resume hint. None → OK.
        self.failure_error: LLMCallError | None = None
        self.recorder = Recorder(
            self.session_id,
            _session_dir,
            parent_session_id=current_session_id.get(),
            resumed_from=resumed_from,
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
        # Skill snapshot + the run's loaded-skill set (shared by reference into
        # every per-message registry so dedup persists across steps). Placement
        # of the index into the opening message happens now, at construction,
        # before any recording — the event is emitted at session start (D9).
        self.skills = skills or {}
        self.loaded_skills = loaded_skills if loaded_skills is not None else set()
        self._rendered_index: RenderedIndex | None = None
        if place_skill_index:
            self._place_skill_index()
        # Usage summaries of delegated subagents spawned by this agent, handed
        # up via ``current_agent_node`` as each ``delegate`` call returns (D3).
        # Each entry already carries its own nested descendants, so a child's
        # rollup folds transitively into this agent's.
        self.child_rollups: list[dict[str, Any]] = []
        self.logger.info(
            "%s initialized with %d messages and %d tools",
            label,
            len(self.messages),
            len(self.tools),
        )

    def _place_skill_index(self) -> None:
        """Append the skill index (and any re-injected bodies) to the opening msg.

        Renders the index from this run's snapshot and appends it after the task
        text of the first user message; for a continuation the previously-loaded
        skills' full bodies follow (D6). Stores the placed :class:`RenderedIndex`
        for the session-start event. A no-op when there are no skills, so the
        opening message stays byte-identical to today.
        """
        if not self.skills:
            return
        block, placed = build_opening_block(self.skills, self.loaded_skills or None)
        if not block:
            return
        for msg in self.messages:
            if msg.get("role") == "user":
                msg["content"] = f"{msg.get('content', '')}\n\n{block}"
                break
        self._rendered_index = placed

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

        self.step_num = self._resume_step
        self.stop_reason = "max_steps"
        self.tool_records = []
        self.handoff_records = []
        self.llm.llm_calls = []
        self.last_prompt_tokens = self._resume_prompt_tokens
        self.handback_report = None
        self.failure_error = None

        t_start = time.monotonic()
        self.recorder.start(self.label, self.llm.model, self.llm.context_window)
        if self._rendered_index is not None:
            # The *offered* record (D9): one skill-index event per session start /
            # continuation, only when an index was actually placed.
            self.recorder.record_skill_index(
                self._rendered_index.names,
                self._rendered_index.chars,
                self._rendered_index.tier,
            )
        _ctx_token = current_session_id.set(self.session_id)
        _rec_token = current_recorder.set(self.recorder)
        _node_token = current_agent_node.set(self)

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
            step_num=self.step_num,
            last_prompt_tokens=self.last_prompt_tokens,
            needs_handback=self.needs_handback,
            skills=self.skills,
            loaded_skills=self.loaded_skills,
        )

        def _spawn_fn() -> list[dict[str, Any]]:
            return self._handle_context_reset(ctx, max_steps, t_start)

        pipeline = build_default_pipeline(
            spawn_fn=_spawn_fn, checkpoint_fn=self._write_checkpoint
        )

        result: list[dict[str, Any]] = []
        try:
            result = pipeline.execute(ctx)
        except KeyboardInterrupt:
            ctx.stop_reason = "aborted"
            self.logger.warning("Agent run aborted by user (KeyboardInterrupt)")
        except LLMCallError as exc:
            # Unrecoverable LLM failure (D6): a first-class stop, not a crash.
            # The last completed step's checkpoint is already on disk (D4), so
            # the run is resumable; the finally block persists session_data.
            ctx.stop_reason = llm_failure_stop_reason(exc.classification)
            self.failure_error = exc
            self.logger.error(
                "Agent run stopped — unrecoverable LLM failure (%s): %s",
                exc.classification,
                exc,
            )
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
            current_agent_node.reset(_node_token)

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
        self.recorder.record_report(content, source=REPORT_SOURCE_FALLBACK)
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
            # Carry the skill snapshot and seed the loaded-set (a copy) so the
            # continuation re-injects the loaded skills' full bodies into its
            # opening message and dedup keeps working post-reset (D6).
            skills=self.skills,
            loaded_skills=set(self.loaded_skills),
        )
        remaining_steps = max_steps - self.step_num
        return continuation.execute(max_steps=max(remaining_steps, 1))

    def add_child_usage(self, summary: dict[str, Any]) -> None:
        """Record a delegated subagent's usage summary (see ``_usage_summary``).

        Called by ``delegate()`` as each subagent finishes, via
        ``current_agent_node``. The summary already carries its own nested
        descendants, so this agent's rollup transitively includes the whole
        delegated subtree (D3).
        """
        self.child_rollups.append(summary)

    def _own_usage_by_kind(self) -> dict[str, dict[str, int]]:
        """Return this agent's own token totals, decomposed per call kind."""
        by_kind: dict[str, dict[str, int]] = {}
        for c in self.llm.llm_calls:
            agg = by_kind.setdefault(
                c.get("kind", "main"),
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            agg["prompt_tokens"] += c["prompt"]
            agg["completion_tokens"] += c["completion"]
            agg["total_tokens"] += c["total"]
        return by_kind

    @staticmethod
    def _sum_totals(by_kind: dict[str, dict[str, int]]) -> dict[str, int]:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for agg in by_kind.values():
            for key in total:
                total[key] += agg[key]
        return total

    def _grand_total(
        self, own_total: dict[str, int], descendants: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Own totals plus every descendant's own grand total (D3)."""
        total = dict(own_total)
        for child in descendants:
            child_total = child["grand_total"]
            for key in total:
                total[key] += child_total[key]
        return total

    def _usage_summary(self, report_source: str | None = None) -> dict[str, Any]:
        """This agent's usage summary: own per-kind totals plus its rollup.

        Returned up through ``delegate()`` (via ``add_child_usage``) so the
        parent accumulates usage without re-reading the child's files (D3).

        ``report_source`` is the delegated report's provenance (verbatim /
        summarizer / fallback), passed by ``delegate()`` for a subagent's own
        summary so the run summary can mark that delegation free or paid
        without re-reading the child's files (D5). ``None`` for the top-level
        agent, which has no report of its own.
        """
        by_kind = self._own_usage_by_kind()
        own_total = self._sum_totals(by_kind)
        return {
            "session_id": self.session_id,
            "elapsed_s": round(self.elapsed_seconds, 3),
            "steps": self.step_num,
            "by_kind": by_kind,
            "descendants": self.child_rollups,
            "grand_total": self._grand_total(own_total, self.child_rollups),
            "report_source": report_source,
        }

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
            rollup=self._usage_summary(),
            resource_rollup=self.recorder.resource_rollup(),
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
            "rollup": self._usage_summary(),
            "context_window": self.llm.context_window,
            "context_reset_threshold": self.context_reset_threshold,
            "tool_records": self.tool_records,
            "handoff_records": self.handoff_records,
            "llm_calls": self.llm.llm_calls,
            "last_message": last_message,
        }
        resource_rollup = self.recorder.resource_rollup()
        if resource_rollup is not None:
            data["resource_rollup"] = resource_rollup
        if self.resumed_from is not None:
            data["resumed_from"] = self.resumed_from
        out = Path(".my_coding_agent") / self.session_id / "session_data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        self.logger.info("Session data saved → %s", out)
        if self.tool_artifacts:
            artifacts_out = out.parent / "tool_artifacts.json"
            artifacts_out.write_text(json.dumps(self.tool_artifacts, indent=2))
            self.logger.info("Tool artifacts saved → %s", artifacts_out)

    def _write_checkpoint(self, ctx: RunContext) -> None:
        """Persist the resume checkpoint at the end of a completed step (D3).

        Passed to the pipeline as ``checkpoint_fn``; the pipeline calls it only
        after a step completes without raising, so a partial step is discarded
        (D4). The write itself is atomic (write-temp + ``os.replace``).
        """
        save_checkpoint(
            self._session_dir,
            Checkpoint(
                session_id=self.session_id,
                step_num=ctx.step_num,
                last_prompt_tokens=ctx.last_prompt_tokens,
                messages=ctx.messages,
            ),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        *,
        api_url: str = OMLX_API_URL,
        api_key: str = OMLX_API_KEY,
        model: str = OMLX_MODEL,
        tools: list[dict[str, Any]] | None = None,
        label: str = "Agent",
        context_reset_threshold: float = 0.75,
        skills: dict[str, Skill] | None = None,
    ) -> AgentNode:
        """Build a fresh AgentNode seeded from a dead session's checkpoint (D5).

        The new node has a brand-new session id and its own clean trace; it loads
        the exact checkpointed conversation and continues the step counter so its
        first LLM call is step N+1. ``checkpoint.session_id`` is recorded as
        ``resumed_from`` — the dead session's files stay immutable.

        ``skills`` is the discovered-skill snapshot so the resumed run's registry
        can service ``use_skill``; its index is *not* re-placed (the checkpointed
        opening message already carries it) via ``place_skill_index=False``.
        """
        return cls(
            api_url=api_url,
            api_key=api_key,
            model=model,
            messages=list(checkpoint.messages),
            tools=tools,
            label=label,
            context_reset_threshold=context_reset_threshold,
            skills=skills,
            place_skill_index=False,
            resumed_from=checkpoint.session_id,
            resume_step=checkpoint.step_num,
            resume_prompt_tokens=checkpoint.last_prompt_tokens,
        )

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

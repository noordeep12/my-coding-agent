"""RunContext — the explicit data contract that flows through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..engine.llm import LLM
    from ..engine.tool_registry.skills import Skill
    from ..observability import Recorder


@dataclass
class RunContext:
    """Carry all mutable run state through the pipeline nodes.

    Immutable config fields are set once at pipeline construction.  Mutable
    state fields are read and written by individual nodes.  Control signals
    (``signal``, ``stop_reason``) are written by nodes and read by
    ``Pipeline.execute`` to decide whether to continue, stop, or reset.
    """

    # --- immutable run config (set once) ---
    session_id: str
    label: str
    max_steps: int
    context_reset_threshold: float
    all_tools: list[dict[str, Any]]
    llm: LLM
    recorder: Recorder
    # "sandboxed" (the #25 OS sandbox is active) or "screened_only" (only the
    # #124 textual refusal gate). Derived once from run config; see
    # ``engine.tool_execution.policy.get_protection_posture``.
    posture: str

    # --- mutable conversation state ---
    messages: list[dict[str, Any]]
    # True when this run owes a hand-back report to a delegating parent; only
    # then does a cutoff trigger report synthesis (standalone runs never pay it).
    needs_handback: bool = False
    step_num: int = 0
    last_prompt_tokens: int = 0
    tool_records: list[dict[str, Any]] = field(default_factory=list)
    tool_artifacts: dict[str, Any] = field(default_factory=dict)
    handoff_records: list[dict[str, Any]] = field(default_factory=list)
    # Discovered-skill snapshot for this run (name → Skill) and the set of skill
    # names loaded so far. The snapshot is stable within a run (D2); the loaded-
    # set is a shared mutable object so `use_skill` dedup persists across steps
    # and rides into a continuation on reset (D5/D6). Both empty when no skills.
    skills: dict[str, Skill] = field(default_factory=dict)
    loaded_skills: set[str] = field(default_factory=set)

    # --- per-step transient state (LLMCallNode writes, downstream nodes read) ---
    last_response: Any = None

    # --- control signals written by nodes, read by the pipeline engine ---
    signal: str = "CONTINUE"  # CONTINUE | STOP | RESET
    stop_reason: str = "max_steps"
    continuation_messages: list[dict[str, Any]] = field(default_factory=list)

    # --- summarizer outputs (ContextSummarizerNode writes, consumers read) ---
    handback_report: str | None = None  # kind "report" → delegate() hand-back
    handoff_content: str | None = None  # kind "handoff" → continuation seed

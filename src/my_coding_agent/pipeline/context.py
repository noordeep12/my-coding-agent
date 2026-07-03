"""RunContext — the explicit data contract that flows through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..engine.llm import LLM
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

    # --- per-step transient state (LLMCallNode writes, downstream nodes read) ---
    last_response: Any = None
    routed_tools: list[dict[str, Any]] = field(default_factory=list)

    # --- control signals written by nodes, read by the pipeline engine ---
    signal: str = "CONTINUE"  # CONTINUE | STOP | RESET
    stop_reason: str = "max_steps"
    continuation_messages: list[dict[str, Any]] = field(default_factory=list)

    # --- summarizer outputs (ContextSummarizerNode writes, consumers read) ---
    handback_report: str | None = None  # kind "report" → delegate() hand-back
    handoff_content: str | None = None  # kind "handoff" → continuation seed

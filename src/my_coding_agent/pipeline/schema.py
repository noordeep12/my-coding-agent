"""Pipeline typed contracts — DAG-level event constants and handoff state."""

from dataclasses import dataclass, field
from datetime import datetime

ROUTER = "router"

# Finish reasons that mark a clean, model-terminated final turn (a non-tool
# assistant message). Shared by FinalizeStepNode's finish check and delegate()'s
# hand-back branch so the two can never drift.
CLEAN_FINISH_REASONS = frozenset({"stop", "exit", "quit"})


@dataclass
class ContextHandoff:
    """Capture state transferred when a run hands off to a fresh context.

    Hold the originating agent's label, step number, and token usage alongside
    the LLM-written ``content`` summarizing progress and remaining work.
    Rendering it as a seed user message and persisting it as a markdown file
    under ``.my_coding_agent/handoffs/`` is builder logic that lives in
    ``pipeline/handoff.py`` (``handoff_to_user_message``, ``save_handoff``).
    """

    agent_label: str
    step_num: int
    prompt_tokens: int
    context_window: int
    content: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    path: str = ""

    @property
    def context_pct(self) -> float:
        """Return prompt tokens as a percent of the context window (0 if unknown)."""
        if not self.context_window:
            return 0.0
        return self.prompt_tokens / self.context_window * 100

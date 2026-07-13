"""Context handoff persistence and rendering — builder logic for ContextHandoff.

Kept separate from ``pipeline/schema.py`` per the schema convention (see
CONTRIBUTE.md §25 and ARCHITECTURE.md): ``schema.py`` holds the
``ContextHandoff`` dataclass shape only; the file I/O and rendering behavior
that operates on it lives here.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import ContextHandoff


def handoff_to_user_message(handoff: ContextHandoff) -> dict[str, Any]:
    """Return a user message that seeds the continuation agent."""
    return {
        "role": "user",
        "content": (
            f"[Context Reset — {handoff.agent_label}, step {handoff.step_num}, "
            f"{handoff.context_pct:.1f}% context used]\n\n"
            f"{handoff.content}"
        ),
    }


def save_handoff(handoff: ContextHandoff, workspace: str = ".") -> str:
    """Persist *handoff* to .my_coding_agent/handoffs/ and return the file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_slug = handoff.agent_label.lower().replace(" ", "_")
    filename = f"{label_slug}_step{handoff.step_num:03d}_{ts}.md"
    out = Path(workspace) / ".my_coding_agent" / "handoffs" / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        f"# Context Handoff\n\n"
        f"- **Agent:** {handoff.agent_label}\n"
        f"- **Step:** {handoff.step_num}\n"
        f"- **Context used:** {handoff.prompt_tokens:,} / {handoff.context_window:,} "
        f"({handoff.context_pct:.1f}%)\n"
        f"- **Created:** {handoff.created_at}\n\n"
        f"---\n\n"
        f"{handoff.content}\n"
    )
    handoff.path = str(out)
    return handoff.path

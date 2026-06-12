"""Context handoff — structured state transfer between context windows."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ContextHandoff:
    agent_label: str
    step_num: int
    prompt_tokens: int
    context_window: int
    content: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    path: str = ""

    @property
    def context_pct(self) -> float:
        if not self.context_window:
            return 0.0
        return self.prompt_tokens / self.context_window * 100

    def to_user_message(self) -> dict:
        """Return a user message that seeds the continuation agent."""
        return {
            "role": "user",
            "content": (
                f"[Context Reset — {self.agent_label}, step {self.step_num}, "
                f"{self.context_pct:.1f}% context used]\n\n"
                f"{self.content}"
            ),
        }

    def save(self, workspace: str = ".") -> str:
        """Persist to .my_coding_agent/handoffs/ and return the file path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_slug = self.agent_label.lower().replace(" ", "_")
        filename = f"{label_slug}_step{self.step_num:03d}_{ts}.md"
        out = Path(workspace) / ".my_coding_agent" / "handoffs" / filename
        out.parent.mkdir(parents=True, exist_ok=True)

        out.write_text(
            f"# Context Handoff\n\n"
            f"- **Agent:** {self.agent_label}\n"
            f"- **Step:** {self.step_num}\n"
            f"- **Context used:** {self.prompt_tokens:,} / {self.context_window:,} "
            f"({self.context_pct:.1f}%)\n"
            f"- **Created:** {self.created_at}\n\n"
            f"---\n\n"
            f"{self.content}\n"
        )
        self.path = str(out)
        return self.path

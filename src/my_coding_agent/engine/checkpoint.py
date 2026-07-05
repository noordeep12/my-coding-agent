"""Engine-owned per-step resume checkpoint (run-resilience D3/D4).

A checkpoint is the exact conversation state at the end of a *completed* step —
the full ``messages`` list plus the minimal counters needed to continue (step
number, last prompt tokens). It is written atomically (write-temp + ``os.replace``)
after each completed step so a run killed mid-step resumes from the end of the
last completed step; the partial step is discarded (D4).

This is deliberately separate from ``events.jsonl``: observability is passive and
must never be read back to drive execution (ARCHITECTURE §25). The checkpoint is
the one file execution owns and reloads on ``--resume``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.exceptions import MyCodingAgentError

CHECKPOINT_FILENAME = "checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(MyCodingAgentError):
    """Raise when a checkpoint is missing or cannot be read for resume."""


@dataclass
class Checkpoint:
    """The exact state needed to resume a run from the end of a completed step.

    ``messages`` is the full conversation (never a summary — zero progress loss);
    ``step_num``/``last_prompt_tokens`` let a resumed run continue as step N+1
    instead of restarting at step 0.
    """

    session_id: str
    step_num: int
    last_prompt_tokens: int
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "step_num": self.step_num,
            "last_prompt_tokens": self.last_prompt_tokens,
            "messages": self.messages,
        }


def checkpoint_path(session_dir: Path) -> Path:
    """Return the checkpoint file path for a session directory."""
    return Path(session_dir) / CHECKPOINT_FILENAME


def save_checkpoint(session_dir: Path, checkpoint: Checkpoint) -> None:
    """Atomically persist *checkpoint* under *session_dir*.

    Writes a sibling temp file then ``os.replace``s it over the final path, so a
    ``kill -9`` mid-write can only ever leave the previous intact checkpoint —
    a reader never observes a torn file (D3).
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_path(session_dir)
    tmp = final.with_suffix(final.suffix + ".tmp")
    tmp.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, final)


def load_checkpoint(session_dir: Path) -> Checkpoint:
    """Load a checkpoint from *session_dir*, or raise a clear ``CheckpointError``.

    Raises:
        CheckpointError: If the checkpoint file is missing or malformed — the
            caller (CLI resume) surfaces this and touches nothing else.
    """
    path = checkpoint_path(session_dir)
    if not path.exists():
        raise CheckpointError(
            f"No checkpoint found at {path}",
            hint="Resume needs a session that ran at least one completed step; "
            "check the session id under .my_coding_agent/.",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"Unreadable checkpoint at {path}: {exc}") from exc
    return Checkpoint(
        session_id=data.get("session_id", Path(session_dir).name),
        step_num=int(data.get("step_num", 0)),
        last_prompt_tokens=int(data.get("last_prompt_tokens", 0)),
        messages=data.get("messages", []),
    )


def find_last_resumable(base_dir: Path) -> str | None:
    """Return the session id of the newest checkpoint under *base_dir*, or None.

    Newest by the checkpoint file's mtime, so ``--resume-last`` targets the run
    that most recently made progress. Sessions without a checkpoint are skipped.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return None
    newest: tuple[float, str] | None = None
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        cp = checkpoint_path(child)
        if not cp.exists():
            continue
        mtime = cp.stat().st_mtime
        if newest is None or mtime > newest[0]:
            newest = (mtime, child.name)
    return newest[1] if newest else None

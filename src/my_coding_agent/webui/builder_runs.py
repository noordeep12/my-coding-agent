"""Launch, monitor, and stop pipeline runs from the Builder tab.

A "run" launched from the Builder is a normal `my-coding-agent` agent run
(reusing `pipeline/dag.py` / `build_default_pipeline` via the standard CLI
entry point, not a parallel execution path) started in a background
subprocess. Progress is derived by tailing the run's own `events.jsonl` —
the same observability stream the Trace Explorer reads — rather than adding
a second reporting channel (D3). Stop sends the process the same
`KeyboardInterrupt` signal a user hits Ctrl-C with; `AgentNode.execute`
already treats that as a clean, resumable stop that still persists the
session (D3's cooperative-stop requirement, via an existing mechanism, not
a new one).

Single-user, one-run-at-a-time (per #153's assumptions): each Builder
session tracks its active runs by generated `run_id`, no cross-process
registry.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

#: How long to wait, after launching, for the new session directory to
#: appear before giving up and reporting the run as unresolved.
_SESSION_DISCOVERY_TIMEOUT_S = 15.0


class RunHandle:
    """Tracks one Builder-launched subprocess and its discovered session."""

    def __init__(self, run_id: str, process: subprocess.Popen, base_dir: Path) -> None:
        self.run_id = run_id
        self.process = process
        self.base_dir = base_dir
        self.launched_at = time.time()
        self.session_id: str | None = None
        self._pre_existing = (
            {p.name for p in base_dir.iterdir() if p.is_dir()}
            if base_dir.is_dir()
            else set()
        )

    def _discover_session_id(self) -> str | None:
        if self.session_id is not None:
            return self.session_id
        if not self.base_dir.is_dir():
            return None
        for child in self.base_dir.iterdir():
            if child.is_dir() and child.name not in self._pre_existing:
                self.session_id = child.name
                return self.session_id
        return None

    def status(self) -> dict[str, Any]:
        session_id = self._discover_session_id()
        exited = self.process.poll() is not None
        events_path = (
            self.base_dir / session_id / "events.jsonl" if session_id else None
        )
        step_num = 0
        last_event_type: str | None = None
        finished = False
        if events_path is not None and events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                event_type = event.get("type")
                if event_type == "llm_call":
                    step_num += 1
                if event_type == "session_end":
                    finished = True
                last_event_type = event_type
        if finished:
            phase = "finished"
        elif exited:
            phase = "stopped" if session_id else "failed"
        elif session_id is None:
            phase = "starting"
        else:
            phase = "running"
        return {
            "run_id": self.run_id,
            "phase": phase,
            "session_id": session_id,
            "step_num": step_num,
            "last_event_type": last_event_type,
        }

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)


class RunRegistry:
    """In-memory registry of Builder-launched runs, keyed by generated run id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunHandle] = {}

    def launch(self, task_prompt: str, max_steps: int, base_dir: Path) -> str:
        run_id = uuid.uuid4().hex[:12]
        process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "my_coding_agent.pipeline.examples.simple",
                "--prompt",
                task_prompt,
                "--max-steps",
                str(max_steps),
            ],
            cwd=Path.cwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._lock:
            self._runs[run_id] = RunHandle(run_id, process, base_dir)
        return run_id

    def get(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(run_id)

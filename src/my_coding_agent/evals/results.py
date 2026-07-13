"""Versioned, self-describing eval run result records and their atomic writer."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..engine.llm import OMLX_MODEL
from .schema import RESULT_SCHEMA_VERSION, EvalScore

RESULTS_ROOT = Path(".my_coding_agent") / "evals"


@dataclass(frozen=True)
class EvalRunResult:
    """A completed eval run: identity, per-case scores, and aggregates.

    Args:
        schema_version: Result-record schema version (see `RESULT_SCHEMA_VERSION`).
        run_id: Unique id for this run.
        timestamp: UTC ISO-8601 timestamp of when the run completed.
        agent_version: `my_coding_agent.__version__` of the agent that ran.
        model: The model name the run used.
        dataset: The case-set/dataset reference this run executed.
        scores: One `EvalScore` per case.
        aggregate_metrics: Run-level metrics (e.g. `{"pass_rate": 0.5}`).
        config_path: Path to the YAML run config that produced this run, if any.
        config_hash: Content hash of that config file, so a result stays
            traceable to the exact configuration version that produced it.
    """

    schema_version: int
    run_id: str
    timestamp: str
    agent_version: str
    model: str
    dataset: str
    scores: list[EvalScore]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    config_path: str | None = None
    config_hash: str | None = None


def build_run_result(
    dataset: str,
    scores: list[EvalScore],
    aggregate_metrics: dict[str, float],
) -> EvalRunResult:
    """Build a run result record with a fresh run id and current timestamp."""
    return EvalRunResult(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(UTC).isoformat(),
        agent_version=__version__,
        model=OMLX_MODEL,
        dataset=dataset,
        scores=scores,
        aggregate_metrics=aggregate_metrics,
    )


def write_run_result(result: EvalRunResult, root: Path = RESULTS_ROOT) -> Path:
    """Atomically write ``result`` under ``root/<run_id>/result.json``.

    Returns:
        The path to the written result directory.
    """
    run_dir = root / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "result.json"
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix="result-", suffix=".json.tmp", dir=run_dir
    )
    try:
        with os.fdopen(tmp_fd, "w") as handle:
            json.dump(asdict(result), handle, indent=2)
        os.replace(tmp_path_str, target)
    finally:
        if os.path.exists(tmp_path_str):
            os.remove(tmp_path_str)
    return run_dir


def load_run_result(run_dir: Path) -> EvalRunResult:
    """Load a written result record back from ``run_dir/result.json``.

    Unknown keys in the record are ignored (forward-compatibility); missing
    keys the reader expects raise via the normal `KeyError`/`TypeError` on
    construction, since a record from a future schema version dropping a
    field this reader needs is not something this reader can recover from.
    """
    raw: dict[str, Any] = json.loads((run_dir / "result.json").read_text())
    scores = [EvalScore(**score) for score in raw["scores"]]
    return EvalRunResult(
        schema_version=raw["schema_version"],
        run_id=raw["run_id"],
        timestamp=raw["timestamp"],
        agent_version=raw["agent_version"],
        model=raw["model"],
        dataset=raw["dataset"],
        scores=scores,
        aggregate_metrics=raw.get("aggregate_metrics", {}),
        config_path=raw.get("config_path"),
        config_hash=raw.get("config_hash"),
    )

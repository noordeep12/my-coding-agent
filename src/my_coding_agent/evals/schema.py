"""Typed contracts for eval cases and scores.

Stdlib-only leaf, like ``engine/sandbox/schema.py`` — kept separate from
``__init__.py`` so the data contract can be imported without pulling in the
loader, scoring, or runner logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Result-record schema version (evals/results.py). Bump on breaking changes;
# prefer additive keys so older readers tolerate newer records.
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalCase:
    """A single eval case: a task prompt plus an outcome check.

    Args:
        id: Stable, unique-within-a-set case identifier.
        task: The task prompt given to the agent.
        scorer: Scorer ref (registry key) that judges the run's outcome.
        expected: Scorer-specific expected/threshold payload.
        dataset: Optional dataset/case-set reference this case belongs to.
        tags: Optional free-form labels for filtering/reporting.
    """

    id: str
    task: str
    scorer: str
    expected: dict[str, Any]
    dataset: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvalScore:
    """A scorer's verdict for one case's run.

    Args:
        case_id: The `EvalCase.id` this score belongs to.
        passed: Overall pass/fail verdict.
        metrics: Numeric metrics keyed by name (e.g. {"match": 1.0}).
        detail: Per-check human-readable detail (e.g. {"reason": "..."}).
        session_id: The session id of the agent run that produced this score,
            when known (additive; `None` for records predating this field).
    """

    case_id: str
    passed: bool
    metrics: dict[str, float]
    detail: dict[str, Any]
    session_id: str | None = None

"""Deterministic trajectory scoring: measure a run's path, not just its answer.

A run can reach a plausible final answer while its internal path was broken —
wrong tool, invalid arguments, an ignored tool failure, or a verification
spiral of redundant re-checks after the task was already done (issue #140).
This module reads a completed run's session directory and scores that path:
process dimensions (tool-selection correctness, argument validity, error
handling) and efficiency dimensions (steps/tokens/wall-clock, redundancy).

Registration against the eval harness's ``Scorer`` contract and scorer
registry (issue #139, ``evals/scoring.py``) lands once that package exists;
until then, ``load_trajectory`` and the ``score_*`` functions below are
usable standalone against any session directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pipeline.anomaly import STREAK_THRESHOLD, trailing_streak
from ..viewer.reader import load_session
from ..viewer.schema import TraceSession

# Tools whose successful call mutates persistent state (the workspace). Used
# by the redundancy dimension to find the last state-changing action. A
# conservative allowlist (CONTRIBUTE.md security standards prefer allowlists
# over denylists) — reads, delegation, and skill lookups don't count even
# though they cost steps, only writes do.
STATE_CHANGING_TOOLS = frozenset({"bash", "write_file"})

_NOT_FOUND_MARKER = "not found"
_WRONG_ARGS_MARKER = "wrong arguments for"


@dataclass(frozen=True)
class CostRollup:
    """A run's cost/resource figures, read from ``session_data.json``.

    Args:
        steps: Number of pipeline steps executed.
        prompt_tokens: Total prompt tokens across all LLM calls.
        completion_tokens: Total completion tokens across all LLM calls.
        total_tokens: Total tokens (prompt + completion).
        elapsed_s: Wall-clock duration of the run, in seconds.
        resource_peaks: Machine-wide resource rollup (``ram_pct``, ``cpu_pct``,
            ``net_bytes``, ``disk_bytes``, ``gpu_pct``), or ``None`` when the
            run wasn't sampled.
    """

    steps: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_s: float
    resource_peaks: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrajectoryTrace:
    """A completed run's trajectory and cost data, ready for dimension scoring.

    Args:
        session: The parsed ``TraceSession`` (decision points, tool-call
            nodes, sub-agent nesting), from ``viewer.reader.load_session``.
        tool_records: The run's tool-call records in call order, from
            ``session_data.json``'s ``tool_records`` — each has ``name``,
            ``args``, ``ok``, and (on failure) ``error``.
        cost: The run's cost/resource rollup.
    """

    session: TraceSession
    tool_records: list[dict[str, Any]] = field(default_factory=list)
    cost: CostRollup = field(default_factory=lambda: CostRollup(0, 0, 0, 0, 0.0, None))


@dataclass(frozen=True)
class DimensionScore:
    """One named, structured measure over a run's trajectory.

    Args:
        name: The dimension's identifier (e.g. ``"tool_selection"``).
        value: Score in ``[0.0, 1.0]``, higher is better.
        detail: Structured detail locating the weakness — which calls
            triggered a penalty, and the raw figures behind ``value``.
    """

    name: str
    value: float
    detail: dict[str, Any]


@dataclass(frozen=True)
class EfficiencyBaseline:
    """Optional reasonable-run limits a case can supply to score cost figures.

    Without a baseline, steps/tokens/wall-clock are reported but not scored
    (a raw step count has no inherent "good" or "bad" without a task-specific
    expectation) — only the baseline-free redundancy signal contributes to
    the efficiency value. A case's expected/threshold payload (issue #139)
    supplies this once the scorer is wired to the harness.

    Args:
        max_steps: Steps at or under which the run scores full marks.
        max_total_tokens: Total tokens at or under which the run scores full marks.
        max_elapsed_s: Wall-clock seconds at or under which the run scores full marks.
    """

    max_steps: int | None = None
    max_total_tokens: int | None = None
    max_elapsed_s: float | None = None


def load_trajectory(session_dir: Path) -> TrajectoryTrace:
    """Load a completed run's trajectory and cost data for scoring.

    Reuses ``viewer.reader.load_session`` for trace parsing rather than
    re-parsing ``events.jsonl`` directly, and reads ``session_data.json`` for
    the flat ``tool_records`` list and cost rollup.

    Args:
        session_dir: The run's session directory (holds ``events.jsonl`` and
            ``session_data.json``).

    Returns:
        The parsed trajectory, ready for dimension scoring.
    """
    session = load_session(session_dir / "events.jsonl")
    session_data = _read_session_data(session_dir)
    tool_records: list[dict[str, Any]] = session_data.get("tool_records", [])
    cost = _build_cost_rollup(session_data)
    return TrajectoryTrace(session=session, tool_records=tool_records, cost=cost)


def _read_session_data(session_dir: Path) -> dict[str, Any]:
    """Read and parse ``session_data.json`` from *session_dir*, or ``{}`` if absent."""
    path = session_dir / "session_data.json"
    if not path.is_file():
        return {}
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _build_cost_rollup(session_data: dict[str, Any]) -> CostRollup:
    """Build a ``CostRollup`` from a parsed ``session_data.json`` dict."""
    total_usage = session_data.get("total_usage", {})
    return CostRollup(
        steps=session_data.get("steps", 0),
        prompt_tokens=total_usage.get("prompt_tokens", 0),
        completion_tokens=total_usage.get("completion_tokens", 0),
        total_tokens=total_usage.get("total_tokens", 0),
        elapsed_s=session_data.get("elapsed_seconds", 0.0),
        resource_peaks=session_data.get("resource_rollup"),
    )


def _failure_reason(record: dict[str, Any]) -> str:
    """Bucket a failed tool-call record into ``not_found``/``wrong_args``/``other``.

    The literal reason isn't persisted as a separate field — it's embedded in
    the record's ``error`` text (``engine/tool_execution``'s ``invoke_tool``),
    so this matches the fixed substrings that text always carries.
    """
    error_text = str(record.get("error", ""))
    if _NOT_FOUND_MARKER in error_text:
        return "not_found"
    if _WRONG_ARGS_MARKER in error_text:
        return "wrong_args"
    return "other"


def score_tool_selection(tool_records: list[dict[str, Any]]) -> DimensionScore:
    """Score whether each call targeted an available (in-toolset) tool.

    Flags ``not_found`` outcomes — a call against a tool name the registry
    doesn't recognize — as off-toolset (issue #140 scenario: "a run that
    called the wrong tool").

    Args:
        tool_records: The run's tool-call records in call order.

    Returns:
        The ``tool_selection`` dimension score; ``1.0`` for an empty run.
    """
    total = len(tool_records)
    if total == 0:
        return DimensionScore(
            "tool_selection", 1.0, {"total_calls": 0, "off_toolset_calls": []}
        )
    off_toolset = [
        i
        for i, record in enumerate(tool_records)
        if not record.get("ok", True) and _failure_reason(record) == "not_found"
    ]
    value = 1.0 - len(off_toolset) / total
    return DimensionScore(
        "tool_selection",
        value,
        {"total_calls": total, "off_toolset_calls": off_toolset},
    )


def score_argument_validity(tool_records: list[dict[str, Any]]) -> DimensionScore:
    """Score the rate of well-formed tool calls.

    Flags ``wrong_args`` outcomes — a call whose arguments didn't match the
    tool's signature (issue #140 scenario: "invalid arguments lower the
    argument-validity dimension").

    Args:
        tool_records: The run's tool-call records in call order.

    Returns:
        The ``argument_validity`` dimension score; ``1.0`` for an empty run.
    """
    total = len(tool_records)
    if total == 0:
        return DimensionScore(
            "argument_validity", 1.0, {"total_calls": 0, "wrong_args_calls": []}
        )
    wrong_args = [
        i
        for i, record in enumerate(tool_records)
        if not record.get("ok", True) and _failure_reason(record) == "wrong_args"
    ]
    value = 1.0 - len(wrong_args) / total
    return DimensionScore(
        "argument_validity",
        value,
        {"total_calls": total, "wrong_args_calls": wrong_args},
    )


def score_error_handling(tool_records: list[dict[str, Any]]) -> DimensionScore:
    """Score whether the run reacted to a failing tool call rather than repeating it.

    Replays ``pipeline.anomaly.trailing_streak`` over successive prefixes of
    *tool_records* — the same signal ``AnomalyDetectNode`` computes live, one
    step at a time — and flags every point a same-signature failure streak
    first reaches ``STREAK_THRESHOLD`` (issue #140 scenario: "an ignored
    repeated failure lowers the error-handling dimension").

    Args:
        tool_records: The run's tool-call records in call order.

    Returns:
        The ``error_handling`` dimension score; ``1.0`` for an empty run.
    """
    total = len(tool_records)
    if total == 0:
        return DimensionScore("error_handling", 1.0, {"ignored_failure_streaks": []})
    streak_events: list[dict[str, Any]] = []
    seen_streak_starts: set[int] = set()
    for i in range(total):
        streak = trailing_streak(tool_records[: i + 1])
        if streak is None:
            continue
        signature, length, member_indexes = streak
        if length >= STREAK_THRESHOLD and member_indexes[0] not in seen_streak_starts:
            seen_streak_starts.add(member_indexes[0])
            streak_events.append(
                {"signature": signature, "length": length, "indexes": member_indexes}
            )
    penalty = min(1.0, sum(e["length"] for e in streak_events) / total)
    value = 1.0 - penalty
    return DimensionScore(
        "error_handling", value, {"ignored_failure_streaks": streak_events}
    )


def score_redundancy(tool_records: list[dict[str, Any]]) -> DimensionScore:
    """Score the run's verification-spiral risk: work after the last mutation.

    Once nothing state-changing (``STATE_CHANGING_TOOLS``) happens again, later
    calls are the "verification spiral" pattern — redundant re-checks after
    the task was already done (issue #140 scenario: "a verification spiral
    scores lower on efficiency").

    Args:
        tool_records: The run's tool-call records in call order.

    Returns:
        The ``redundancy`` dimension score; ``1.0`` for an empty run.
    """
    total = len(tool_records)
    if total == 0:
        return DimensionScore(
            "redundancy",
            1.0,
            {"trailing_calls_after_last_mutation": 0, "total_calls": 0},
        )
    last_mutation = None
    for i, record in enumerate(tool_records):
        if record.get("name") in STATE_CHANGING_TOOLS:
            last_mutation = i
    trailing = total - (last_mutation + 1) if last_mutation is not None else total
    value = 1.0 - trailing / total
    return DimensionScore(
        "redundancy",
        value,
        {"trailing_calls_after_last_mutation": trailing, "total_calls": total},
    )


def score_efficiency(
    cost: CostRollup,
    tool_records: list[dict[str, Any]],
    baseline: EfficiencyBaseline | None = None,
) -> DimensionScore:
    """Score efficiency from the cost rollup and the redundancy signal.

    Steps/tokens/wall-clock are always reported in ``detail``; they only
    contribute to ``value`` when *baseline* supplies a limit to score them
    against (a raw step count alone has no inherent "good" or "bad"). The
    redundancy sub-score always contributes, since it's self-contained.

    Args:
        cost: The run's cost rollup.
        tool_records: The run's tool-call records in call order.
        baseline: Optional reasonable-run limits for steps/tokens/wall-clock.

    Returns:
        The ``efficiency`` dimension score.
    """
    redundancy = score_redundancy(tool_records)
    sub_scores = [redundancy.value]
    detail: dict[str, Any] = {
        "steps": cost.steps,
        "prompt_tokens": cost.prompt_tokens,
        "completion_tokens": cost.completion_tokens,
        "total_tokens": cost.total_tokens,
        "elapsed_s": cost.elapsed_s,
        "redundancy": redundancy.detail,
    }
    if baseline is not None:
        for label, actual, limit in (
            ("steps", cost.steps, baseline.max_steps),
            ("total_tokens", cost.total_tokens, baseline.max_total_tokens),
            ("elapsed_s", cost.elapsed_s, baseline.max_elapsed_s),
        ):
            if limit is None or limit <= 0:
                continue
            ratio_score = 1.0 if actual <= limit else limit / actual
            sub_scores.append(ratio_score)
            detail[f"{label}_vs_baseline"] = {
                "actual": actual,
                "baseline": limit,
                "score": ratio_score,
            }
    value = sum(sub_scores) / len(sub_scores)
    return DimensionScore("efficiency", value, detail)

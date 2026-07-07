"""Run a case set against the agent in isolated workspaces and score it."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..engine.agent import AgentNode
from ..engine.tool_registry import ToolRegistry, tool
from .schema import EvalCase, EvalScore
from .scoring import RunResult, UnknownScorerError, resolve_scorer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a coding assistant completing a single evaluation task. Use "
    "tools to get things done, then state your final answer clearly."
)


def _build_tools() -> list[dict[str, Any]]:
    import inspect

    names = [
        name
        for name, _ in inspect.getmembers(ToolRegistry, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    return [tool(getattr(ToolRegistry, name)) for name in names]


def _final_output(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _run_case_in_workspace(case: EvalCase) -> RunResult:
    """Run one case's agent turn inside a fresh temp workspace, isolated from
    the real repo and from other cases (cwd + `.my_coding_agent/` session dir
    both resolve under the temp dir)."""
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="mca-eval-") as workspace:
        try:
            os.chdir(workspace)
            agent = AgentNode(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": case.task},
                ],
                tools=_build_tools(),
                label=f"Eval[{case.id}]",
            )
            try:
                messages = agent.execute()
            except Exception:
                logger.exception("case %s: agent run errored", case.id)
                return RunResult(
                    final_output="",
                    session_dir=Path(workspace) / ".my_coding_agent" / agent.session_id,
                    session_id=agent.session_id,
                    errored=True,
                )
            return RunResult(
                final_output=_final_output(messages),
                session_dir=Path(workspace) / ".my_coding_agent" / agent.session_id,
                session_id=agent.session_id,
                errored=agent.failure_error is not None,
            )
        finally:
            os.chdir(original_cwd)


def run_case_set(cases: list[EvalCase]) -> tuple[list[EvalScore], dict[str, float]]:
    """Run every case in isolation, score it, and aggregate run-level metrics.

    A case whose agent run errors is recorded as a failed case (via the
    scorer, given `RunResult.errored=True`); the runner still continues to
    the remaining cases rather than crashing.

    Returns:
        Per-case scores (one per case, in input order) and aggregate metrics
        (currently `pass_rate`).
    """
    scores: list[EvalScore] = []
    for case in cases:
        run_result = _run_case_in_workspace(case)
        try:
            scorer = resolve_scorer(case.scorer)
        except UnknownScorerError as exc:
            logger.error("case %s: %s", case.id, exc)
            scores.append(
                EvalScore(
                    case_id=case.id,
                    passed=False,
                    metrics={},
                    detail={"reason": str(exc)},
                )
            )
            continue
        scores.append(scorer.score(case, run_result))

    pass_rate = (
        sum(1 for score in scores if score.passed) / len(scores) if scores else 0.0
    )
    return scores, {"pass_rate": pass_rate}

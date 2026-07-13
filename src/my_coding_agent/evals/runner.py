"""Run a case set against the agent in isolated workspaces and score it."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..engine.tool_registry import ToolRegistry, tool
from ..pipeline.nodes.agent import AgentNode
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


def _run_and_score_case(case: EvalCase) -> EvalScore:
    """Run one case's agent turn inside a fresh temp workspace and score it.

    Isolated from the real repo and from other cases (cwd + `.my_coding_agent/`
    session dir both resolve under the temp dir). Scoring happens before the
    temp workspace is torn down (the `with` block still open) so a scorer
    that reads the run's trace via `RunResult.session_dir` (e.g. the
    trajectory scorer, #140) finds `events.jsonl`/`session_data.json` still
    on disk — the workspace and its session dir don't outlive this call.
    """
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
                run_result = RunResult(
                    final_output=_final_output(messages),
                    session_dir=Path(workspace) / ".my_coding_agent" / agent.session_id,
                    session_id=agent.session_id,
                    errored=agent.failure_error is not None,
                )
            except Exception:
                logger.exception("case %s: agent run errored", case.id)
                run_result = RunResult(
                    final_output="",
                    session_dir=Path(workspace) / ".my_coding_agent" / agent.session_id,
                    session_id=agent.session_id,
                    errored=True,
                )
        finally:
            os.chdir(original_cwd)

        try:
            scorer = resolve_scorer(case.scorer)
        except UnknownScorerError as exc:
            logger.error("case %s: %s", case.id, exc)
            return EvalScore(
                case_id=case.id,
                passed=False,
                metrics={},
                detail={"reason": str(exc)},
                session_id=run_result.session_id,
            )
        return replace(scorer.score(case, run_result), session_id=run_result.session_id)


def run_case_set(cases: list[EvalCase]) -> tuple[list[EvalScore], dict[str, float]]:
    """Run every case in isolation, score it, and aggregate run-level metrics.

    A case whose agent run errors is recorded as a failed case (via the
    scorer, given `RunResult.errored=True`); the runner still continues to
    the remaining cases rather than crashing.

    Returns:
        Per-case scores (one per case, in input order) and aggregate metrics
        (currently `pass_rate`).
    """
    scores = [_run_and_score_case(case) for case in cases]

    pass_rate = (
        sum(1 for score in scores if score.passed) / len(scores) if scores else 0.0
    )
    return scores, {"pass_rate": pass_rate}

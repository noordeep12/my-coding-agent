"""Scorer contract, baseline scorer, and the scorer registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import EvalCase, EvalScore


class UnknownScorerError(Exception):
    """Raised when a case references a scorer ref not in the registry."""

    def __init__(self, scorer_ref: str) -> None:
        super().__init__(
            f"unknown scorer {scorer_ref!r} — no scorer registered under that "
            "ref; check the case's `scorer` field or register the scorer"
        )
        self.scorer_ref = scorer_ref


@dataclass(frozen=True)
class RunResult:
    """A completed agent run: its final output plus a trace handle.

    Args:
        final_output: The text content of the agent's final assistant message.
        session_dir: Directory holding the run's `events.jsonl`/`session_data.json`.
        session_id: The run's session id.
        errored: True if the agent run itself failed (not the case outcome).
    """

    final_output: str
    session_dir: Path
    session_id: str
    errored: bool = False


class Scorer(Protocol):
    """A scorer judges one case's completed run and returns a structured score."""

    def score(self, case: EvalCase, run_result: RunResult) -> EvalScore:
        """Return this case's verdict, metrics, and per-check detail."""
        ...


class ExactMatchScorer:
    """Baseline deterministic scorer: exact/threshold check on final output.

    ``case.expected`` supports:
        - ``contains``: substring that must appear in the final output.
        - ``equals``: the final output must equal this string exactly.
    At least one of the two must be present.
    """

    def score(self, case: EvalCase, run_result: RunResult) -> EvalScore:
        if run_result.errored:
            return EvalScore(
                case_id=case.id,
                passed=False,
                metrics={"match": 0.0},
                detail={"reason": "agent run errored"},
            )

        output = run_result.final_output
        equals = case.expected.get("equals")
        contains = case.expected.get("contains")

        if equals is not None:
            passed = output == equals
            detail = {"reason": "equals check", "expected": equals, "actual": output}
        elif contains is not None:
            passed = contains in output
            detail = {
                "reason": "contains check",
                "expected": contains,
                "actual": output,
            }
        else:
            passed = False
            detail = {"reason": "case.expected has neither 'equals' nor 'contains'"}

        return EvalScore(
            case_id=case.id,
            passed=passed,
            metrics={"match": 1.0 if passed else 0.0},
            detail=detail,
        )


_REGISTRY: dict[str, Scorer] = {
    "exact_match": ExactMatchScorer(),
}


def resolve_scorer(scorer_ref: str) -> Scorer:
    """Look up a scorer by ref, raising `UnknownScorerError` if unregistered."""
    scorer = _REGISTRY.get(scorer_ref)
    if scorer is None:
        raise UnknownScorerError(scorer_ref)
    return scorer


def register_scorer(scorer_ref: str, scorer: Scorer) -> None:
    """Register a scorer under ``scorer_ref``, the registry's single extension point."""
    _REGISTRY[scorer_ref] = scorer

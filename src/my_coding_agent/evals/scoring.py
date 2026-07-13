"""Scorer contract, baseline scorer, and the scorer registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..utils.exceptions import MyCodingAgentError
from .schema import EvalCase, EvalScore


class UnknownScorerError(MyCodingAgentError):
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


def list_scorer_refs() -> list[str]:
    """Return the registered scorer refs, sorted."""
    return sorted(_REGISTRY)


#: Required `expected` keys per built-in scorer ref, used only for pre-save
#: validation in the UI config layer — the scorers themselves already fail
#: gracefully (as a failed case, not a crash) on a missing key at run time.
#: `exact_match` needs exactly one of two alternative keys, so it is checked
#: separately rather than listed here.
_REQUIRED_EXPECTED_KEYS: dict[str, tuple[str, ...]] = {
    "judge": ("rubric", "pass_threshold"),
}


def validate_expected(scorer_ref: str, expected: dict) -> list[str]:
    """Check ``expected`` against the selected scorer's known shape.

    Returns a list of human-readable problems (empty if valid). Only checks
    the built-in scorers' documented required keys; an unrecognized scorer
    ref (e.g. one registered outside this module) is not validated here.
    """
    if scorer_ref not in _REGISTRY:
        return [f"unknown scorer {scorer_ref!r}"]

    if scorer_ref == "exact_match":
        if "equals" not in expected and "contains" not in expected:
            return ["exact_match requires 'equals' or 'contains' in expected"]
        return []

    required = _REQUIRED_EXPECTED_KEYS.get(scorer_ref, ())
    missing = [key for key in required if key not in expected]
    return [f"{scorer_ref} requires '{key}' in expected" for key in missing]

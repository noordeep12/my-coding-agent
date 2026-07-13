"""Rubric-based LLM judge: structured verdicts over a declared rubric, plus
calibration against human labels so the judge's trustworthiness is measured,
not assumed.

A rubric is a disk artifact (JSON) a case references — never a free-form
"is this good?" prompt. The judge makes one bounded model call per case,
tagged with the ``judge`` call kind (``engine.llm.schema.CALL_KIND_JUDGE``) so
its cost is captured like any other call, and returns a ``JudgeVerdict``
carrying a rationale per criterion so bias (verbosity/confidence preference)
is inspectable rather than hidden behind a single number.

``JudgeScorer`` adapts a ``JudgeVerdict`` to the eval harness's ``Scorer``
contract (``evals.scoring``) and is registered under the ``"judge"`` ref by
this module's import (see ``evals/__init__.py``).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.llm import LLM
from ..engine.llm import parsing as llm_parsing
from ..engine.llm.schema import CALL_KIND_JUDGE
from ..utils.exceptions import MyCodingAgentError
from .schema import EvalCase, EvalScore
from .scoring import RunResult, register_scorer

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Default cap on the judge's completion — a verdict is a small structured
# JSON payload, not free-form prose.
DEFAULT_JUDGE_MAX_TOKENS = 800


class RubricError(MyCodingAgentError):
    """A rubric artifact is missing, malformed, or fails validation."""


class JudgeError(MyCodingAgentError):
    """The judge's response could not be parsed into a structured verdict."""


# --- Rubric ------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    """One judged dimension: a name, what it means, and score anchors."""

    name: str
    description: str
    anchors: dict[int, str]  # score -> what that score means for this criterion


@dataclass(frozen=True)
class Rubric:
    """A declared, versionable set of criteria and a shared score scale."""

    name: str
    scale_min: int
    scale_max: int
    criteria: tuple[Criterion, ...]


def load_rubric(path: Path) -> Rubric:
    """Load a rubric from a JSON artifact.

    Args:
        path: Path to the rubric JSON file.

    Returns:
        The parsed, validated ``Rubric``.

    Raises:
        RubricError: The file is missing, not valid JSON, or fails structural
            validation (missing keys, empty criteria, an out-of-scale anchor,
            or an inverted scale). Never falls back to a free-form judge.
    """
    try:
        raw = path.read_text()
    except OSError as exc:
        raise RubricError(f"Rubric file not found or unreadable: {path}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RubricError(f"Rubric {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise RubricError(f"Rubric {path} must be a JSON object")

    return _rubric_from_dict(data, source=str(path))


def _rubric_from_dict(data: dict[str, Any], *, source: str) -> Rubric:
    try:
        name = data["name"]
        scale = data["scale"]
        scale_min = int(scale["min"])
        scale_max = int(scale["max"])
        raw_criteria = data["criteria"]
    except KeyError as exc:
        raise RubricError(f"Rubric {source} missing required field: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise RubricError(f"Rubric {source} has a malformed 'scale': {exc}") from exc

    if not isinstance(name, str) or not name:
        raise RubricError(f"Rubric {source} 'name' must be a non-empty string")
    if scale_min >= scale_max:
        raise RubricError(
            f"Rubric {source} scale.min ({scale_min}) must be < scale.max ({scale_max})"
        )
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise RubricError(f"Rubric {source} must declare at least one criterion")

    criteria = tuple(
        _criterion_from_dict(
            c, index=i, scale_min=scale_min, scale_max=scale_max, source=source
        )
        for i, c in enumerate(raw_criteria)
    )
    return Rubric(
        name=name, scale_min=scale_min, scale_max=scale_max, criteria=criteria
    )


def _criterion_from_dict(
    data: Any, *, index: int, scale_min: int, scale_max: int, source: str
) -> Criterion:
    if not isinstance(data, dict):
        raise RubricError(f"Rubric {source} criteria[{index}] must be an object")
    try:
        name = data["name"]
        description = data["description"]
        raw_anchors = data["anchors"]
    except KeyError as exc:
        raise RubricError(
            f"Rubric {source} criteria[{index}] missing required field: {exc}"
        ) from exc

    if not isinstance(name, str) or not name:
        raise RubricError(
            f"Rubric {source} criteria[{index}] 'name' must be a non-empty string"
        )
    if not isinstance(description, str) or not description:
        raise RubricError(
            f"Rubric {source} criteria[{index}] 'description' must be non-empty"
        )
    if not isinstance(raw_anchors, dict) or not raw_anchors:
        raise RubricError(
            f"Rubric {source} criteria[{index}] must declare at least one anchor"
        )

    anchors: dict[int, str] = {}
    for key, text in raw_anchors.items():
        try:
            score = int(key)
        except (TypeError, ValueError) as exc:
            raise RubricError(
                f"Rubric {source} criteria[{index}] anchor key {key!r} "
                "is not an integer"
            ) from exc
        if not (scale_min <= score <= scale_max):
            raise RubricError(
                f"Rubric {source} criteria[{index}] anchor {score} is outside "
                f"the rubric scale [{scale_min}, {scale_max}]"
            )
        anchors[score] = text

    return Criterion(name=name, description=description, anchors=anchors)


# --- Judge scorer --------------------------------------------------------------


@dataclass(frozen=True)
class CriterionVerdict:
    """The judge's score and rationale for one criterion."""

    name: str
    score: int
    rationale: str


@dataclass(frozen=True)
class JudgeVerdict:
    """A structured judge verdict: a score + rationale per criterion, plus an
    overall score/rationale. Carrying the per-criterion rationale is what
    makes the judge's bias (verbosity/confidence preference) inspectable
    rather than hidden behind one number.
    """

    criteria: tuple[CriterionVerdict, ...]
    overall_score: float
    overall_rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for embedding in a persisted result record."""
        return {
            "criteria": [
                {"name": c.name, "score": c.score, "rationale": c.rationale}
                for c in self.criteria
            ],
            "overall_score": self.overall_score,
            "overall_rationale": self.overall_rationale,
        }


def _build_judge_prompt(rubric: Rubric, task: str, output: str) -> str:
    criteria_lines = []
    for c in rubric.criteria:
        anchor_lines = "\n".join(
            f"    {score}: {text}" for score, text in sorted(c.anchors.items())
        )
        criteria_lines.append(f"- {c.name}: {c.description}\n{anchor_lines}")
    criteria_block = "\n".join(criteria_lines)
    criterion_names = ", ".join(f'"{c.name}"' for c in rubric.criteria)
    criteria_shape = ", ".join(
        f'"{c.name}": {{"score": <int>, "rationale": "<why>"}}' for c in rubric.criteria
    )

    return (
        "/no_think\n"
        "You are grading an AI agent's output against a fixed rubric. Score each "
        "criterion independently using only the stated scale and anchors — do not "
        "invent criteria and do not let a confident or verbose tone raise a score "
        "that the content does not earn.\n\n"
        f"Scale: integers from {rubric.scale_min} to {rubric.scale_max}.\n\n"
        f"Criteria:\n{criteria_block}\n\n"
        f"Task given to the agent:\n{task}\n\n"
        f"Agent's output:\n{output}\n\n"
        "Respond with ONLY a JSON object of this exact shape, no other text:\n"
        "{\n"
        f'  "criteria": {{{criteria_shape}}},\n'
        '  "overall_score": <number>,\n'
        '  "overall_rationale": "<why>"\n'
        "}\n"
        f"(criteria keys must be exactly: {criterion_names})"
    )


def _extract_json_candidate(content: str) -> str:
    """Pick the JSON candidate substring out of a judge response.

    Prefers the last fenced ```json block (the model's actual answer, which
    follows any illustrative example it may have echoed earlier in its
    reasoning), falls back to the outermost ``{...}`` span, and raises if
    neither is present rather than guessing at a boundary.
    """
    fenced: list[str] = _FENCED_JSON_RE.findall(content)
    if fenced:
        return fenced[-1]

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeError("Judge response contains no JSON object")
    return content[start : end + 1]


def _parse_judge_response(content: str, rubric: Rubric) -> JudgeVerdict:
    cleaned = _THINK_RE.sub("", content).strip()
    candidate = _extract_json_candidate(cleaned)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Judge response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise JudgeError("Judge response JSON must be an object")

    try:
        raw_criteria = data["criteria"]
        overall_score = float(data["overall_score"])
        overall_rationale = str(data["overall_rationale"])
    except KeyError as exc:
        raise JudgeError(f"Judge response missing required field: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise JudgeError(f"Judge response has a malformed field: {exc}") from exc

    if not isinstance(raw_criteria, dict):
        raise JudgeError("Judge response 'criteria' must be an object")

    expected_names = {c.name for c in rubric.criteria}
    got_names = set(raw_criteria)
    if got_names != expected_names:
        raise JudgeError(
            f"Judge response criteria {sorted(got_names)} do not match rubric "
            f"criteria {sorted(expected_names)}"
        )

    verdicts = tuple(
        _criterion_verdict_from_entry(raw_criteria[c.name], criterion=c, rubric=rubric)
        for c in rubric.criteria
    )

    return JudgeVerdict(
        criteria=verdicts,
        overall_score=overall_score,
        overall_rationale=overall_rationale,
    )


def _criterion_verdict_from_entry(
    entry: Any, *, criterion: Criterion, rubric: Rubric
) -> CriterionVerdict:
    name = criterion.name
    if not isinstance(entry, dict):
        raise JudgeError(f"Judge response criterion {name!r} must be an object")
    try:
        score = int(entry["score"])
        rationale = str(entry["rationale"])
    except KeyError as exc:
        raise JudgeError(
            f"Judge response criterion {name!r} missing field: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise JudgeError(
            f"Judge response criterion {name!r} has a malformed field: {exc}"
        ) from exc
    if not (rubric.scale_min <= score <= rubric.scale_max):
        raise JudgeError(
            f"Judge response criterion {name!r} score {score} is outside "
            f"the rubric scale [{rubric.scale_min}, {rubric.scale_max}]"
        )
    return CriterionVerdict(name=name, score=score, rationale=rationale)


def score_with_judge(
    llm: Any,
    rubric: Rubric,
    task: str,
    output: str,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> JudgeVerdict:
    """Make one bounded judge call and return a structured verdict.

    Args:
        llm: An ``engine.llm.LLM``-shaped client exposing ``chat_completion``.
        rubric: The declared rubric to grade against.
        task: The task prompt the agent was given.
        output: The agent's final output for this run.
        max_tokens: Cap on the judge's completion.

    Returns:
        A ``JudgeVerdict`` with a score + rationale per criterion.

    Raises:
        JudgeError: The judge's response was truncated before completion, or
            could not be parsed into a verdict matching the rubric. Never
            silently degrades to a guessed score.
    """
    prompt = _build_judge_prompt(rubric, task, output)
    resp = llm.chat_completion(
        [{"role": "user", "content": prompt}],
        tools=[],
        kind=CALL_KIND_JUDGE,
        max_tokens=max_tokens,
    )
    if llm_parsing.extract_finish_reason(resp) == "length":
        raise JudgeError(
            "Judge response was truncated (finish_reason=length) before "
            "completing its verdict; raise max_tokens rather than trusting "
            "a partial response"
        )
    content = llm_parsing.extract_message(resp).get("content") or ""
    return _parse_judge_response(content, rubric)


# --- Calibration against human labels ------------------------------------------


@dataclass(frozen=True)
class LabelledCase:
    """One human-labelled example for calibration: a task/output pair and the
    human's score for a single criterion (or an overall label)."""

    task: str
    output: str
    human_score: int


@dataclass(frozen=True)
class CalibrationResult:
    """Agreement between the judge and human labels, plus the reliability
    verdict against a stated threshold."""

    agreement: float
    threshold: float
    reliable: bool
    judge_scores: tuple[int, ...]
    human_scores: tuple[int, ...]


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for agreement between two raters over the same items.

    Args:
        a: First rater's scores (categorical).
        b: Second rater's scores, same length and order as ``a``.

    Returns:
        Kappa in roughly [-1, 1]; 1.0 is perfect agreement, 0.0 is chance-level.

    Raises:
        ValueError: ``a`` and ``b`` differ in length, or either is empty.
    """
    if len(a) != len(b):
        raise ValueError(f"Score lists must be the same length: {len(a)} != {len(b)}")
    if not a:
        raise ValueError("Cannot compute agreement over an empty set of scores")

    n = len(a)
    observed_agreement = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    categories = set(a) | set(b)
    freq_a = {c: a.count(c) / n for c in categories}
    freq_b = {c: b.count(c) / n for c in categories}
    expected_agreement = sum(freq_a[c] * freq_b[c] for c in categories)

    if expected_agreement == 1.0:
        return 1.0 if observed_agreement == 1.0 else 0.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


# Below this agreement with human labels, the judge is flagged unreliable
# rather than silently trusted.
DEFAULT_RELIABILITY_THRESHOLD = 0.6


def calibrate(
    llm: Any,
    rubric: Rubric,
    labelled_cases: list[LabelledCase],
    *,
    criterion_name: str,
    threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
) -> CalibrationResult:
    """Score a human-labelled set with the judge and report agreement.

    Args:
        llm: The LLM client the judge scores through.
        rubric: The rubric to grade against.
        labelled_cases: Human-labelled task/output/score examples.
        criterion_name: Which rubric criterion's judge score to compare
            against ``LabelledCase.human_score``.
        threshold: Minimum Cohen's kappa for the judge to be considered
            reliable.

    Returns:
        A ``CalibrationResult`` carrying the agreement figure and whether it
        clears ``threshold`` — an unreliable judge is flagged, not hidden.
    """
    judge_scores = []
    human_scores = []
    for case in labelled_cases:
        verdict = score_with_judge(llm, rubric, case.task, case.output)
        criterion = next(
            (c for c in verdict.criteria if c.name == criterion_name), None
        )
        if criterion is None:
            raise JudgeError(
                f"Judge verdict has no criterion {criterion_name!r} "
                "to calibrate against"
            )
        judge_scores.append(criterion.score)
        human_scores.append(case.human_score)

    agreement = cohens_kappa(judge_scores, human_scores)
    return CalibrationResult(
        agreement=agreement,
        threshold=threshold,
        reliable=agreement >= threshold,
        judge_scores=tuple(judge_scores),
        human_scores=tuple(human_scores),
    )


# --- Scorer registration --------------------------------------------------


class JudgeScorer:
    """Adapts the rubric judge to the eval harness's ``Scorer`` contract.

    ``case.expected`` must carry:
        - ``rubric``: path to the rubric JSON artifact.
        - ``pass_threshold``: minimum ``overall_score`` to pass the case.

    ``case.expected`` may optionally carry:
        - ``max_tokens``: override for the judge's completion cap (default
          ``DEFAULT_JUDGE_MAX_TOKENS``). A verbose reasoning model can exceed
          the default while producing a multi-criterion verdict; raising this
          per-case is preferable to a truncated response being discarded.

    An errored agent run, a missing/malformed rubric, or an unparseable
    judge response are all recorded as a failed case with the reason in
    ``EvalScore.detail`` — never a crashed run.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm if llm is not None else LLM()

    def score(self, case: EvalCase, run_result: RunResult) -> EvalScore:
        if run_result.errored:
            return EvalScore(
                case_id=case.id,
                passed=False,
                metrics={},
                detail={"reason": "agent run errored"},
            )

        rubric_ref = case.expected.get("rubric")
        if not rubric_ref:
            return EvalScore(
                case_id=case.id,
                passed=False,
                metrics={},
                detail={"reason": "case.expected missing 'rubric'"},
            )
        threshold = case.expected.get("pass_threshold")
        if threshold is None:
            return EvalScore(
                case_id=case.id,
                passed=False,
                metrics={},
                detail={"reason": "case.expected missing 'pass_threshold'"},
            )

        max_tokens = int(case.expected.get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS))

        try:
            rubric = load_rubric(Path(rubric_ref))
            verdict = score_with_judge(
                self._llm,
                rubric,
                case.task,
                run_result.final_output,
                max_tokens=max_tokens,
            )
        except (RubricError, JudgeError) as exc:
            return EvalScore(
                case_id=case.id, passed=False, metrics={}, detail={"reason": str(exc)}
            )

        metrics = {"overall_score": verdict.overall_score}
        metrics.update({f"{c.name}_score": float(c.score) for c in verdict.criteria})
        return EvalScore(
            case_id=case.id,
            passed=verdict.overall_score >= float(threshold),
            metrics=metrics,
            detail=verdict.to_dict(),
        )


register_scorer("judge", JudgeScorer())

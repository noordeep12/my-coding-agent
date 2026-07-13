"""Evaluation: binds a reusable RunConfig to a reusable EvalConfig.

An Evaluation is the missing primitive between *how the agent runs* (a
RunConfig: model/prompt/params/tools) and *what its output is judged
against* (an EvalConfig: an ordered Rule -> Check tree). Running an
Evaluation drives the agent once through the existing case-runner path
(`runner.py`) and scores the produced output against every Check via the
existing scorer registry (`scoring.py`), writing the result through the
existing result writer (`results.py`) — this module adds no new execution
or scoring engine, only the binding + persistence + orchestration.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..engine.agent import AgentNode
from ..engine.llm import LLM
from ..utils.exceptions import MyCodingAgentError
from .results import EvalRunResult, build_run_result, write_run_result
from .runner import _build_tools, _final_output
from .schema import EvalCase, EvalScore
from .scoring import RunResult, UnknownScorerError, resolve_scorer

logger = logging.getLogger(__name__)

DEFAULT_EVALUATIONS_DIR = Path(".my_coding_agent/evals/evaluations")
DEFAULT_RUN_CONFIGS_DIR = Path(".my_coding_agent/evals/run_configs")
DEFAULT_EVAL_CONFIGS_DIR = Path(".my_coding_agent/evals/eval_configs")

_DEFAULT_SYSTEM_PROMPT = (
    "You are a coding assistant completing a single evaluation task. Use "
    "tools to get things done, then state your final answer clearly."
)


class EvaluationError(MyCodingAgentError):
    """Base exception for evaluation-management operations."""


class NotFoundError(EvaluationError):
    """Raised when an Evaluation/RunConfig/EvalConfig id has no file on disk."""


class UnresolvedReferenceError(EvaluationError):
    """Raised when an Evaluation references a config id that fails to resolve."""


class UnknownEvaluatorError(EvaluationError):
    """Raised when a Check names an evaluator type with no registered scorer."""


def new_id() -> str:
    """Return a fresh stable id, matching the shape of a run id."""
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """A single scored assertion within a Rule.

    Args:
        id: Stable check identifier.
        name: Human-readable check name.
        description: Optional free-text description.
        method: Evaluation method label (e.g. "equals", "contains", "rubric").
        input: The input fed to the evaluator (evaluator-specific shape).
        expected: The expected value the evaluator checks against.
        evaluator: Evaluator type; resolves to a registered scorer
            (`exact_match` / `trajectory` / `judge`).
        threshold: Pass threshold the evaluator's score is compared against.
    """

    id: str
    name: str
    description: str = ""
    method: str = ""
    input: Any = None
    expected: Any = None
    evaluator: str = "exact_match"
    threshold: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "method": self.method,
            "input": self.input,
            "expected": self.expected,
            "evaluator": self.evaluator,
            "threshold": self.threshold,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Check:
        return Check(
            id=str(data.get("id") or new_id()),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            method=str(data.get("method", "")),
            input=data.get("input"),
            expected=data.get("expected"),
            evaluator=str(data.get("evaluator", "exact_match")),
            threshold=float(data.get("threshold", 1.0)),
        )


@dataclass(frozen=True)
class Rule:
    """An ordered group of Checks within an EvalConfig."""

    id: str
    name: str
    description: str = ""
    checks: tuple[Check, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "checks": [c.to_dict() for c in self.checks],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Rule:
        return Rule(
            id=str(data.get("id") or new_id()),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            checks=tuple(Check.from_dict(c) for c in data.get("checks") or []),
        )


@dataclass(frozen=True)
class EvalConfig:
    """A persisted, reusable Rule -> Check tree."""

    id: str
    name: str
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "rules": [r.to_dict() for r in self.rules],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EvalConfig:
        return EvalConfig(
            id=str(data.get("id") or new_id()),
            name=str(data.get("name", "")),
            rules=tuple(Rule.from_dict(r) for r in data.get("rules") or []),
        )

    def checks(self) -> list[Check]:
        return [check for rule in self.rules for check in rule.checks]


@dataclass(frozen=True)
class RunConfig:
    """A persisted, reusable execution configuration for the agent pipeline."""

    id: str
    name: str
    description: str = ""
    agent: str = ""
    model: str = ""
    provider: str = ""
    system_prompt: str = ""
    user_prompt_template: str = ""
    context_template: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    tools_enabled: bool = True
    tool_config: dict[str, Any] = field(default_factory=dict)
    memory_config: dict[str, Any] = field(default_factory=dict)
    retrieval_config: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent": self.agent,
            "model": self.model,
            "provider": self.provider,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "context_template": self.context_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "extra_params": self.extra_params,
            "tools_enabled": self.tools_enabled,
            "tool_config": self.tool_config,
            "memory_config": self.memory_config,
            "retrieval_config": self.retrieval_config,
            "env_vars": self.env_vars,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunConfig:
        return RunConfig(
            id=str(data.get("id") or new_id()),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            agent=str(data.get("agent", "")),
            model=str(data.get("model", "")),
            provider=str(data.get("provider", "")),
            system_prompt=str(data.get("system_prompt", "")),
            user_prompt_template=str(data.get("user_prompt_template", "")),
            context_template=str(data.get("context_template", "")),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            top_p=data.get("top_p"),
            extra_params=dict(data.get("extra_params") or {}),
            tools_enabled=bool(data.get("tools_enabled", True)),
            tool_config=dict(data.get("tool_config") or {}),
            memory_config=dict(data.get("memory_config") or {}),
            retrieval_config=dict(data.get("retrieval_config") or {}),
            env_vars=dict(data.get("env_vars") or {}),
        )


@dataclass(frozen=True)
class LastRun:
    """Summary of an Evaluation's most recent run."""

    run_id: str
    verdict: str
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "verdict": self.verdict, "at": self.at}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LastRun:
        return LastRun(
            run_id=str(data["run_id"]), verdict=str(data["verdict"]), at=str(data["at"])
        )


@dataclass(frozen=True)
class Evaluation:
    """A first-class, persisted binding of one RunConfig to one EvalConfig."""

    id: str
    name: str
    summary: str = ""
    run_config_id: str = ""
    eval_config_id: str = ""
    last_run: LastRun | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "run_config_id": self.run_config_id,
            "eval_config_id": self.eval_config_id,
            "last_run": self.last_run.to_dict() if self.last_run else None,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Evaluation:
        last_run_raw = data.get("last_run")
        return Evaluation(
            id=str(data.get("id") or new_id()),
            name=str(data.get("name", "")),
            summary=str(data.get("summary", "")),
            run_config_id=str(data.get("run_config_id", "")),
            eval_config_id=str(data.get("eval_config_id", "")),
            last_run=LastRun.from_dict(last_run_raw) if last_run_raw else None,
        )


# --------------------------------------------------------------------------
# Evaluator resolution
# --------------------------------------------------------------------------


def validate_evaluator(evaluator: str) -> None:
    """Raise `UnknownEvaluatorError` unless `evaluator` resolves to a scorer."""
    try:
        resolve_scorer(evaluator)
    except UnknownScorerError as exc:
        raise UnknownEvaluatorError(str(exc)) from exc


def validate_eval_config(eval_config: EvalConfig) -> None:
    """Raise `UnknownEvaluatorError` for the first Check with an unknown evaluator."""
    for check in eval_config.checks():
        validate_evaluator(check.evaluator)


# --------------------------------------------------------------------------
# Persistence — one JSON file per object, mirroring evals/cases.py's pattern.
# --------------------------------------------------------------------------


def _path(object_id: str, base_dir: Path) -> Path:
    return base_dir / f"{object_id}.json"


def _write(base_dir: Path, object_id: str, data: dict[str, Any]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = _path(object_id, base_dir)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _read(base_dir: Path, object_id: str) -> dict[str, Any]:
    path = _path(object_id, base_dir)
    if not path.exists():
        raise NotFoundError(f"No object '{object_id}' found under {base_dir}")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _delete(base_dir: Path, object_id: str) -> None:
    path = _path(object_id, base_dir)
    if not path.exists():
        raise NotFoundError(f"No object '{object_id}' found under {base_dir}")
    path.unlink()


def _list(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )


# -- RunConfig ---------------------------------------------------------------


def create_run_config(
    data: dict[str, Any], *, base_dir: Path = DEFAULT_RUN_CONFIGS_DIR
) -> RunConfig:
    run_config = RunConfig.from_dict({**data, "id": new_id()})
    _write(base_dir, run_config.id, run_config.to_dict())
    return run_config


def get_run_config(
    run_config_id: str, *, base_dir: Path = DEFAULT_RUN_CONFIGS_DIR
) -> RunConfig:
    return RunConfig.from_dict(_read(base_dir, run_config_id))


def list_run_configs(*, base_dir: Path = DEFAULT_RUN_CONFIGS_DIR) -> list[RunConfig]:
    return [RunConfig.from_dict(json.loads(p.read_text())) for p in _list(base_dir)]


def update_run_config(
    run_config_id: str,
    data: dict[str, Any],
    *,
    base_dir: Path = DEFAULT_RUN_CONFIGS_DIR,
) -> RunConfig:
    existing = get_run_config(run_config_id, base_dir=base_dir)
    merged = {**existing.to_dict(), **data, "id": run_config_id}
    run_config = RunConfig.from_dict(merged)
    _write(base_dir, run_config.id, run_config.to_dict())
    return run_config


def delete_run_config(
    run_config_id: str, *, base_dir: Path = DEFAULT_RUN_CONFIGS_DIR
) -> None:
    _delete(base_dir, run_config_id)


# -- EvalConfig ----------------------------------------------------------


def create_eval_config(
    data: dict[str, Any], *, base_dir: Path = DEFAULT_EVAL_CONFIGS_DIR
) -> EvalConfig:
    eval_config = EvalConfig.from_dict({**data, "id": new_id()})
    validate_eval_config(eval_config)
    _write(base_dir, eval_config.id, eval_config.to_dict())
    return eval_config


def get_eval_config(
    eval_config_id: str, *, base_dir: Path = DEFAULT_EVAL_CONFIGS_DIR
) -> EvalConfig:
    return EvalConfig.from_dict(_read(base_dir, eval_config_id))


def list_eval_configs(*, base_dir: Path = DEFAULT_EVAL_CONFIGS_DIR) -> list[EvalConfig]:
    return [EvalConfig.from_dict(json.loads(p.read_text())) for p in _list(base_dir)]


def update_eval_config(
    eval_config_id: str,
    data: dict[str, Any],
    *,
    base_dir: Path = DEFAULT_EVAL_CONFIGS_DIR,
) -> EvalConfig:
    existing = get_eval_config(eval_config_id, base_dir=base_dir)
    merged = {**existing.to_dict(), **data, "id": eval_config_id}
    eval_config = EvalConfig.from_dict(merged)
    validate_eval_config(eval_config)
    _write(base_dir, eval_config.id, eval_config.to_dict())
    return eval_config


def delete_eval_config(
    eval_config_id: str, *, base_dir: Path = DEFAULT_EVAL_CONFIGS_DIR
) -> None:
    _delete(base_dir, eval_config_id)


# -- Evaluation ------------------------------------------------------------


def _validate_references(
    run_config_id: str,
    eval_config_id: str,
    *,
    run_configs_dir: Path,
    eval_configs_dir: Path,
) -> None:
    if not _path(run_config_id, run_configs_dir).exists():
        raise UnresolvedReferenceError(
            f"run_config_id '{run_config_id}' does not resolve to a stored RunConfig"
        )
    if not _path(eval_config_id, eval_configs_dir).exists():
        raise UnresolvedReferenceError(
            f"eval_config_id '{eval_config_id}' does not resolve to a stored EvalConfig"
        )


def create_evaluation(
    data: dict[str, Any],
    *,
    base_dir: Path = DEFAULT_EVALUATIONS_DIR,
    run_configs_dir: Path = DEFAULT_RUN_CONFIGS_DIR,
    eval_configs_dir: Path = DEFAULT_EVAL_CONFIGS_DIR,
) -> Evaluation:
    evaluation = Evaluation.from_dict({**data, "id": new_id(), "last_run": None})
    _validate_references(
        evaluation.run_config_id,
        evaluation.eval_config_id,
        run_configs_dir=run_configs_dir,
        eval_configs_dir=eval_configs_dir,
    )
    _write(base_dir, evaluation.id, evaluation.to_dict())
    return evaluation


def get_evaluation(
    evaluation_id: str, *, base_dir: Path = DEFAULT_EVALUATIONS_DIR
) -> Evaluation:
    return Evaluation.from_dict(_read(base_dir, evaluation_id))


def list_evaluations(*, base_dir: Path = DEFAULT_EVALUATIONS_DIR) -> list[Evaluation]:
    return [Evaluation.from_dict(json.loads(p.read_text())) for p in _list(base_dir)]


def update_evaluation(
    evaluation_id: str,
    data: dict[str, Any],
    *,
    base_dir: Path = DEFAULT_EVALUATIONS_DIR,
    run_configs_dir: Path = DEFAULT_RUN_CONFIGS_DIR,
    eval_configs_dir: Path = DEFAULT_EVAL_CONFIGS_DIR,
) -> Evaluation:
    existing = get_evaluation(evaluation_id, base_dir=base_dir)
    merged = {**existing.to_dict(), **data, "id": evaluation_id}
    evaluation = Evaluation.from_dict(merged)
    _validate_references(
        evaluation.run_config_id,
        evaluation.eval_config_id,
        run_configs_dir=run_configs_dir,
        eval_configs_dir=eval_configs_dir,
    )
    _write(base_dir, evaluation.id, evaluation.to_dict())
    return evaluation


def delete_evaluation(
    evaluation_id: str, *, base_dir: Path = DEFAULT_EVALUATIONS_DIR
) -> None:
    _delete(base_dir, evaluation_id)


# --------------------------------------------------------------------------
# Run execution — reuses runner.py's isolated-workspace path and the
# existing scorer registry; writes results through results.py.
# --------------------------------------------------------------------------


def _run_config_task(run_config: RunConfig) -> str:
    parts = [
        p for p in (run_config.user_prompt_template, run_config.context_template) if p
    ]
    return "\n\n".join(parts) if parts else run_config.description or run_config.name


def _check_expected(check: Check) -> dict[str, Any]:
    """Build the `EvalCase.expected` dict a registered scorer expects.

    A Check's `expected`/`input`/`threshold` map onto the same shape a
    dataset case's `expected` already carries per scorer
    (`exact_match`: `equals`/`contains`; `judge`: `rubric`/`pass_threshold`;
    `trajectory`: optional `pass_threshold`/baseline keys) — no new scoring
    semantics, only translating the Check's fields onto the existing ones.
    """
    if isinstance(check.expected, dict):
        expected = dict(check.expected)
    else:
        key = (
            check.method
            if check.method in ("equals", "contains", "rubric")
            else "equals"
        )
        expected = {key: check.expected}
    expected.setdefault("pass_threshold", check.threshold)
    return expected


def _run_agent(
    run_config: RunConfig, evaluation_id: str, *, llm_client: LLM | None = None
) -> tuple[str, RunResult]:
    """Run the agent once per the RunConfig inside an isolated temp workspace.

    ``llm_client``, if given, supplies the connection (api_url/api_key/model/
    timeout) the agent's LLM client is built from — e.g. the interface's
    resolved Admin settings — instead of `AgentNode`'s own env-var defaults.

    Caller must consume the returned `RunResult` (which may reference
    `session_dir`) before the temp workspace this was invoked under tears
    down — see `run_evaluation` below.
    """
    system_prompt = run_config.system_prompt or _DEFAULT_SYSTEM_PROMPT
    task = _run_config_task(run_config)
    agent_kwargs: dict[str, Any] = {}
    if llm_client is not None:
        agent_kwargs = {
            "api_url": llm_client.api_url,
            "api_key": llm_client.api_key,
            "model": llm_client.model,
        }
    agent = AgentNode(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ],
        tools=_build_tools() if run_config.tools_enabled else [],
        label=f"Evaluation[{evaluation_id}]",
        **agent_kwargs,
    )
    if llm_client is not None:
        agent.llm.timeout = llm_client.timeout
        agent.llm.setup_session()
    try:
        messages = agent.execute()
        run_result = RunResult(
            final_output=_final_output(messages),
            session_dir=Path.cwd() / ".my_coding_agent" / agent.session_id,
            session_id=agent.session_id,
            errored=agent.failure_error is not None,
        )
    except Exception:
        logger.exception("evaluation %s: agent run errored", evaluation_id)
        run_result = RunResult(
            final_output="",
            session_dir=Path.cwd() / ".my_coding_agent" / agent.session_id,
            session_id=agent.session_id,
            errored=True,
        )
    return task, run_result


def _score_checks(
    checks: list[Check], task: str, run_result: RunResult
) -> list[EvalScore]:
    scores = []
    for check in checks:
        case = EvalCase(
            id=check.id,
            task=task,
            scorer=check.evaluator,
            expected=_check_expected(check),
        )
        try:
            scorer = resolve_scorer(check.evaluator)
        except UnknownScorerError as exc:
            scores.append(
                EvalScore(
                    case_id=check.id,
                    passed=False,
                    metrics={},
                    detail={"reason": str(exc)},
                )
            )
            continue
        scores.append(scorer.score(case, run_result))
    return scores


def run_evaluation(
    evaluation: Evaluation,
    *,
    run_configs_dir: Path = DEFAULT_RUN_CONFIGS_DIR,
    eval_configs_dir: Path = DEFAULT_EVAL_CONFIGS_DIR,
    evaluations_dir: Path = DEFAULT_EVALUATIONS_DIR,
    results_root: Path | None = None,
    run_id: str | None = None,
    llm_client: LLM | None = None,
) -> EvalRunResult:
    """Run `evaluation`'s RunConfig through the agent and score every Check.

    Runs the pipeline once, in the current working directory (unlike
    `runner.py`'s isolated-temp-workspace pattern for dataset cases), so the
    run's session lands under the project's real `.my_coding_agent/`
    directory — visible to the Traces tab — and any relative path a Check's
    `expected` carries (e.g. a judge rubric path) resolves against the real
    project root instead of a throwaway directory. Scores every Check across
    every Rule against that single run's output, via the existing scorer
    registry. Writes the result through `results.py` and updates the
    Evaluation's `last_run`.

    ``run_id``, if given, overrides the freshly-generated run id
    `build_run_result` assigns — so a caller that already handed the run id
    back to a client (e.g. the web UI's fire-and-poll endpoint, which returns
    a run id before this function has finished) can find the written result
    under the id it was promised.

    ``llm_client``, if given, is forwarded to `_run_agent` so the run's agent
    is built from that connection instead of env-var defaults (see
    `_run_agent`).
    """
    run_config = get_run_config(evaluation.run_config_id, base_dir=run_configs_dir)
    eval_config = get_eval_config(evaluation.eval_config_id, base_dir=eval_configs_dir)
    checks = eval_config.checks()

    if not checks:
        scores: list[EvalScore] = []
        aggregate_metrics = {
            "pass_rate": 0.0,
            "checks_total": 0.0,
            "nothing_to_score": 1.0,
        }
        verdict = "no_checks"
    else:
        task, run_result = _run_agent(run_config, evaluation.id, llm_client=llm_client)
        scores = _score_checks(checks, task, run_result)
        pass_rate = sum(1 for s in scores if s.passed) / len(scores)
        aggregate_metrics = {"pass_rate": pass_rate, "checks_total": float(len(scores))}
        verdict = "pass" if pass_rate == 1.0 else "fail"

    result = build_run_result(f"evaluation:{evaluation.id}", scores, aggregate_metrics)
    if run_id is not None:
        result = replace(result, run_id=run_id)
    if results_root is None:
        write_run_result(result)
    else:
        write_run_result(result, root=results_root)

    updated = replace(
        evaluation,
        last_run=LastRun(run_id=result.run_id, verdict=verdict, at=result.timestamp),
    )
    _write(evaluations_dir, updated.id, updated.to_dict())
    return result

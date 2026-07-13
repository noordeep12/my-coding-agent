"""Load, validate, and execute a single declarative YAML run configuration.

A run config file is the complete, self-contained definition of one eval/
pipeline run: LLM connection, prompts, run parameters, and evaluation
criteria. It maps onto the existing `RunConfig`/`Check` domain models
(`evaluation.py`) and executes through the existing orchestration (a single
real-cwd agent run scored via the scorer registry, plus/instead of existing
case and dataset references run through the existing case runner) — this
module adds no new execution or scoring engine, only load, validate, and
dispatch.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..engine.agent import DEFAULT_MAX_STEPS, AgentNode
from ..engine.llm import LLM
from ..utils.exceptions import MyCodingAgentError
from . import cases as evals_cases
from . import datasets as evals_datasets
from .evaluation import (
    _DEFAULT_SYSTEM_PROMPT,
    Check,
    RunConfig,
    _check_expected,
)
from .results import EvalRunResult, build_run_result, write_run_result
from .runner import _build_tools, _final_output, run_case_set
from .schema import EvalCase, EvalScore
from .scoring import RunResult, UnknownScorerError, resolve_scorer

logger = logging.getLogger(__name__)

_TOP_LEVEL_KEYS = {"llm", "run", "evaluation"}
_LLM_KEYS = {"api_url", "model", "api_key_env", "api_key", "timeout"}
_RUN_KEYS = {"system_prompt", "task", "max_steps"}
_EVAL_KEYS = {"checks", "cases", "dataset"}
_CHECK_KEYS = {
    "id",
    "name",
    "description",
    "method",
    "input",
    "expected",
    "evaluator",
    "threshold",
}

#: Field -> (env var name, documented default). Mirrors `webui/admin.py`'s
#: resolution order (config value -> env var -> default), with "config" in
#: place of "saved settings" — reads the environment live, not the
#: import-time-frozen `engine.llm` constants.
_ENV_FIELDS: dict[str, tuple[str, str]] = {
    "api_url": ("OMLX_API_URL", "http://127.0.0.1:8321/v1"),
    "model": ("OMLX_MODEL", "Qwen3.6-35B-A3B-6bit"),
}
_DEFAULT_API_KEY_ENV = "OMLX_API_KEY"  # pragma: allowlist secret
_DEFAULT_API_KEY = "changeme"  # pragma: allowlist secret
_DEFAULT_TIMEOUT = 30.0


class ConfigError(MyCodingAgentError):
    """Base exception for config-driven run loading/validation."""


class ConfigValidationError(ConfigError):
    """Raised when a config file fails validation; carries every problem found."""

    def __init__(self, path: Path, problems: list[str]) -> None:
        super().__init__(
            f"invalid config {path}: " + "; ".join(problems),
        )
        self.path = path
        self.problems = problems


@dataclass(frozen=True)
class LoadedRunConfig:
    """A fully-parsed, validated run config file.

    Args:
        run_config: The run's connection + prompt + run-parameter fields,
            shaped as the existing `RunConfig` model.
        checks: Inline checks scored against the config's own single run.
        case_ids: Existing case ids (`evals/cases.py`) to run through the
            isolated case runner, alongside/instead of `checks`.
        dataset_id: An existing dataset id (`evals/datasets.py`) to run
            through the isolated case runner, alongside/instead of `checks`.
        content_hash: SHA-256 hex digest of the config file's raw bytes.
    """

    run_config: RunConfig
    checks: tuple[Check, ...]
    case_ids: tuple[str, ...]
    dataset_id: str | None
    content_hash: str
    llm_section: dict[str, Any]


def _resolve_field(config_value: Any, env_var: str, default: str) -> str:
    if config_value not in (None, ""):
        return str(config_value)
    return os.environ.get(env_var, default)


def build_llm_client(llm_section: dict[str, Any]) -> LLM:
    """Build an `LLM` client from a parsed config's `llm` section.

    Each field resolves config value -> environment variable -> documented
    default. The API key is never read from the config directly — only
    `api_key_env` (an environment variable *name*) is accepted; validation
    rejects a raw `api_key` value before this is ever called.
    """
    api_url = _resolve_field(llm_section.get("api_url"), *_ENV_FIELDS["api_url"])
    model = _resolve_field(llm_section.get("model"), *_ENV_FIELDS["model"])
    api_key_env = str(llm_section.get("api_key_env") or _DEFAULT_API_KEY_ENV)
    api_key = os.environ.get(api_key_env, _DEFAULT_API_KEY)
    timeout_raw = llm_section.get("timeout")
    timeout = float(timeout_raw) if timeout_raw else _DEFAULT_TIMEOUT
    return LLM(api_url=api_url, api_key=api_key, model=model, timeout=timeout)


def _validate_section(
    data: dict[str, Any], key: str, allowed_keys: set[str], problems: list[str]
) -> dict[str, Any]:
    section = data.get(key) or {}
    if not isinstance(section, dict):
        problems.append(f"'{key}' section must be a mapping")
        return {}
    unknown = set(section) - allowed_keys
    if unknown:
        problems.append(f"unknown '{key}' key(s): {', '.join(sorted(unknown))}")
    return section


def _parse_checks(raw_checks: Any, problems: list[str]) -> tuple[Check, ...]:
    if raw_checks in (None, ""):
        return ()
    if not isinstance(raw_checks, list):
        problems.append("'evaluation.checks' must be a list")
        return ()

    checks: list[Check] = []
    for index, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, dict):
            problems.append(f"'evaluation.checks[{index}]' must be a mapping")
            continue
        unknown = set(raw_check) - _CHECK_KEYS
        if unknown:
            problems.append(
                f"unknown key(s) in 'evaluation.checks[{index}]': "
                f"{', '.join(sorted(unknown))}"
            )
        evaluator = str(raw_check.get("evaluator", "exact_match"))
        try:
            resolve_scorer(evaluator)
        except UnknownScorerError as exc:
            problems.append(f"'evaluation.checks[{index}]': {exc}")
            continue
        check_id = raw_check.get("id") or f"check-{index}"
        checks.append(Check.from_dict({**raw_check, "id": check_id}))
    return tuple(checks)


def _parse_max_steps(run_section: dict[str, Any], problems: list[str]) -> int:
    max_steps_raw = run_section.get("max_steps")
    if not max_steps_raw:
        return DEFAULT_MAX_STEPS
    try:
        return int(max_steps_raw)
    except (TypeError, ValueError):
        problems.append(f"'run.max_steps' must be an integer, got {max_steps_raw!r}")
        return DEFAULT_MAX_STEPS


def _parse_evaluation_refs(
    eval_section: dict[str, Any], checks: tuple[Check, ...], problems: list[str]
) -> tuple[tuple[str, ...], str | None]:
    case_ids = tuple(str(c) for c in (eval_section.get("cases") or []))
    dataset_id_raw = eval_section.get("dataset")
    dataset_id = str(dataset_id_raw) if dataset_id_raw else None
    if not checks and not case_ids and not dataset_id:
        problems.append(
            "'evaluation' must declare at least one of: checks, cases, dataset"
        )
    return case_ids, dataset_id


def load_config_file(path: Path) -> LoadedRunConfig:
    """Load and validate ``path`` into a `LoadedRunConfig`.

    Raises:
        ConfigValidationError: On malformed YAML, an unknown top-level or
            section key, a missing required field, an unknown evaluator
            reference, or a raw `api_key` value — collected and reported
            together rather than one at a time.
    """
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ConfigValidationError(path, [f"cannot read file: {exc}"]) from exc

    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(path, [f"malformed YAML: {exc}"]) from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigValidationError(path, ["config root must be a mapping"])

    problems: list[str] = []
    unknown_top = set(data) - _TOP_LEVEL_KEYS
    if unknown_top:
        problems.append(f"unknown top-level key(s): {', '.join(sorted(unknown_top))}")

    llm_section = _validate_section(data, "llm", _LLM_KEYS, problems)
    if "api_key" in llm_section:
        problems.append(
            "'llm.api_key' is not allowed — use 'api_key_env: <VAR_NAME>' to "
            "reference an environment variable instead of a raw secret value"
        )

    run_section = _validate_section(data, "run", _RUN_KEYS, problems)
    task = run_section.get("task")
    if not task:
        problems.append("'run.task' is required")

    eval_section = _validate_section(data, "evaluation", _EVAL_KEYS, problems)
    checks = _parse_checks(eval_section.get("checks"), problems)
    case_ids, dataset_id = _parse_evaluation_refs(eval_section, checks, problems)
    max_steps = _parse_max_steps(run_section, problems)

    if problems:
        raise ConfigValidationError(path, problems)

    run_config = RunConfig.from_dict(
        {
            "id": "config-run",
            "name": path.stem,
            "system_prompt": run_section.get("system_prompt", ""),
            "user_prompt_template": task,
            "extra_params": {"max_steps": max_steps},
        }
    )

    return LoadedRunConfig(
        run_config=run_config,
        checks=checks,
        case_ids=case_ids,
        dataset_id=dataset_id,
        content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        llm_section=llm_section,
    )


def _run_agent(
    run_config: RunConfig, label: str, llm_client: LLM
) -> tuple[str, RunResult]:
    """Run the agent once, in the real cwd, using ``llm_client``'s connection.

    Mirrors `evaluation._run_agent`, but also forwards `run_config`'s
    `max_steps` (carried in `extra_params`) to `AgentNode.execute`, which the
    reused helper does not need since evaluations don't configure it.
    """
    system_prompt = run_config.system_prompt or _DEFAULT_SYSTEM_PROMPT
    task = run_config.user_prompt_template
    max_steps = int(run_config.extra_params.get("max_steps", DEFAULT_MAX_STEPS))
    agent = AgentNode(
        api_url=llm_client.api_url,
        api_key=llm_client.api_key,
        model=llm_client.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ],
        tools=_build_tools(),
        label=label,
    )
    agent.llm.timeout = llm_client.timeout
    agent.llm.setup_session()
    try:
        messages = agent.execute(max_steps=max_steps)
        run_result = RunResult(
            final_output=_final_output(messages),
            session_dir=Path.cwd() / ".my_coding_agent" / agent.session_id,
            session_id=agent.session_id,
            errored=agent.failure_error is not None,
        )
    except Exception:
        logger.exception("config run %s: agent run errored", label)
        run_result = RunResult(
            final_output="",
            session_dir=Path.cwd() / ".my_coding_agent" / agent.session_id,
            session_id=agent.session_id,
            errored=True,
        )
    return task, run_result


def _score_checks(
    checks: tuple[Check, ...], task: str, run_result: RunResult
) -> list[EvalScore]:
    scores = []
    for check in checks:
        case = EvalCase(
            id=check.id,
            task=task,
            scorer=check.evaluator,
            expected=_check_expected(check),
        )
        scorer = resolve_scorer(check.evaluator)
        scores.append(scorer.score(case, run_result))
    return scores


def _resolve_case_refs(case_ids: tuple[str, ...]) -> list[EvalCase]:
    if not case_ids:
        return []
    all_cases = evals_cases.load_case_set(evals_datasets.DEFAULT_CASES_DIR)
    by_id = {case.id: case for case in all_cases}
    resolved = []
    for case_id in case_ids:
        case = by_id.get(case_id)
        if case is None:
            logger.warning(
                "config run: case '%s' not found under %s",
                case_id,
                evals_datasets.DEFAULT_CASES_DIR,
            )
            continue
        resolved.append(case)
    return resolved


def execute_from_config(
    path: Path, *, results_root: Path | None = None
) -> tuple[EvalRunResult, str]:
    """Load, validate, and execute ``path`` end to end.

    Runs the config's inline `checks` (if any) as a single real-cwd agent
    run scored against every check, plus any `cases`/`dataset` references
    run through the existing isolated case runner — aggregating every score
    into one result record, stamped with the config's path and content hash.

    Returns:
        The written result record and its verdict (`"pass"`/`"fail"`).

    Raises:
        ConfigValidationError: If the config fails validation; no agent run
            or session directory is created.
    """
    loaded = load_config_file(path)

    scores: list[EvalScore] = []

    if loaded.checks:
        llm_client = build_llm_client(loaded.llm_section)
        task, run_result = _run_agent(
            loaded.run_config, f"ConfigRun[{path.name}]", llm_client
        )
        scores.extend(_score_checks(loaded.checks, task, run_result))

    referenced_cases = list(_resolve_case_refs(loaded.case_ids))
    if loaded.dataset_id is not None:
        # This is the project's own evals.datasets.load_dataset (a local
        # JSON-file store), not huggingface datasets.load_dataset -- no download.
        dataset = evals_datasets.load_dataset(dataset_id=loaded.dataset_id)  # nosec B615
        referenced_cases.extend(evals_datasets.resolve_cases(dataset))
    if referenced_cases:
        case_scores, _ = run_case_set(referenced_cases)
        scores.extend(case_scores)

    pass_rate = sum(1 for s in scores if s.passed) / len(scores) if scores else 0.0
    aggregate_metrics = {"pass_rate": pass_rate, "checks_total": float(len(scores))}
    verdict = "pass" if scores and pass_rate == 1.0 else "fail"

    result = build_run_result(f"config:{path}", scores, aggregate_metrics)
    result = replace(result, config_path=str(path), config_hash=loaded.content_hash)
    if results_root is None:
        write_run_result(result)
    else:
        write_run_result(result, root=results_root)

    return result, verdict

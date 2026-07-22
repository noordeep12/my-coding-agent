"""Load, validate, and execute a single declarative YAML run configuration.

A run config file is the complete, self-contained definition of one eval/
pipeline run: LLM connection, prompts, run parameters, and inline evaluation
checks. It maps onto the existing `RunConfig`/`Check` domain models
(`evaluation.py`) and executes through the existing orchestration (a single
real-cwd agent run scored via the scorer registry) — this module adds no new
execution or scoring engine, only load, validate, and dispatch.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..engine.llm import LLM
from ..pipeline.graph import Pipeline
from ..pipeline.nodes.agent import DEFAULT_MAX_STEPS, AgentNode, PipelineBuilder
from ..pipeline.nodes.prompt_stage import PromptStageNode
from ..pipeline.schema import Transition
from ..utils.exceptions import MyCodingAgentError
from .evaluation import (
    _DEFAULT_SYSTEM_PROMPT,
    Check,
    RunConfig,
    _check_expected,
)
from .results import EvalRunResult, build_run_result, write_run_result
from .runner import _build_tools, _final_output
from .schema import EvalCase, EvalScore
from .scoring import RunResult, UnknownScorerError, resolve_scorer

logger = logging.getLogger(__name__)

_TOP_LEVEL_KEYS = {"llm", "run", "evaluation", "pipeline"}
_LLM_KEYS = {"api_url", "model", "api_key_env", "api_key", "timeout"}
_RUN_KEYS = {"system_prompt", "task", "max_steps"}
_EVAL_KEYS = {"checks"}
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
_PIPELINE_KEYS = {"nodes", "transitions"}
_PIPELINE_NODE_KEYS = {"name", "prompt", "accept_if_contains"}
_PIPELINE_TRANSITION_KEYS = {"source", "target", "max_rounds"}

#: Field -> (env var name, documented default). Resolution order is
#: config value -> env var -> default; reads the environment live, not the
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
        content_hash: SHA-256 hex digest of the config file's raw bytes.
        llm_section: The parsed `llm` mapping, forwarded to `build_llm_client`.
        pipeline_nodes: Declared workflow-graph stages (issue #228) from an
            optional `pipeline` section, or `None` when the config uses the
            standard single-agent pipeline (the default).
        pipeline_transitions: Declared conditional transitions between
            `pipeline_nodes`, or `None` alongside them.
    """

    run_config: RunConfig
    checks: tuple[Check, ...]
    content_hash: str
    llm_section: dict[str, Any]
    pipeline_nodes: tuple[PromptStageNode, ...] | None = None
    pipeline_transitions: tuple[Transition, ...] | None = None


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


def _parse_pipeline_node_specs(
    raw_nodes: list[Any], problems: list[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse `pipeline.nodes` into `(node_names, node_specs)`; see `_parse_pipeline`."""
    node_names: list[str] = []
    node_specs: list[dict[str, Any]] = []
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            problems.append(f"'pipeline.nodes[{index}]' must be a mapping")
            continue
        unknown_node = set(raw_node) - _PIPELINE_NODE_KEYS
        if unknown_node:
            problems.append(
                f"unknown key(s) in 'pipeline.nodes[{index}]': "
                f"{', '.join(sorted(unknown_node))}"
            )
        name = raw_node.get("name")
        prompt = raw_node.get("prompt")
        if not name:
            problems.append(f"'pipeline.nodes[{index}].name' is required")
            continue
        if not prompt:
            problems.append(f"'pipeline.nodes[{index}].prompt' is required")
            continue
        if name in node_names:
            problems.append(f"duplicate pipeline node name: {name!r}")
            continue
        node_names.append(name)
        node_specs.append(
            {
                "name": name,
                "prompt": prompt,
                "accept_if_contains": raw_node.get("accept_if_contains"),
            }
        )
    return node_names, node_specs


def _parse_pipeline_transitions(
    raw_transitions: Any, node_names: list[str], problems: list[str]
) -> tuple[list[Transition], dict[str, str]]:
    """Parse `pipeline.transitions` into `(transitions, jump_target_by_source)`."""
    if not isinstance(raw_transitions, list):
        if raw_transitions:
            problems.append("'pipeline.transitions' must be a list")
        raw_transitions = []

    transitions: list[Transition] = []
    jump_target_by_source: dict[str, str] = {}
    for index, raw_t in enumerate(raw_transitions):
        if not isinstance(raw_t, dict):
            problems.append(f"'pipeline.transitions[{index}]' must be a mapping")
            continue
        unknown_t = set(raw_t) - _PIPELINE_TRANSITION_KEYS
        if unknown_t:
            problems.append(
                f"unknown key(s) in 'pipeline.transitions[{index}]': "
                f"{', '.join(sorted(unknown_t))}"
            )
        source = raw_t.get("source")
        target = raw_t.get("target")
        if not source or not target:
            problems.append(
                f"'pipeline.transitions[{index}]' requires 'source' and 'target'"
            )
            continue
        if source not in node_names:
            problems.append(
                f"'pipeline.transitions[{index}].source' {source!r} is not a "
                "declared node"
            )
        if target not in node_names:
            problems.append(
                f"'pipeline.transitions[{index}].target' {target!r} is not a "
                "declared node"
            )
        transitions.append(
            Transition(source=source, target=target, max_rounds=raw_t.get("max_rounds"))
        )
        jump_target_by_source[source] = target
    return transitions, jump_target_by_source


def _build_pipeline_nodes(
    node_specs: list[dict[str, Any]],
    jump_target_by_source: dict[str, str],
    problems: list[str],
) -> list[PromptStageNode]:
    """Build `PromptStageNode`s, wiring each decision node's jump target."""
    nodes: list[PromptStageNode] = []
    for spec in node_specs:
        accept = spec["accept_if_contains"]
        jump_target = jump_target_by_source.get(spec["name"]) if accept else None
        if accept and jump_target is None:
            problems.append(
                f"pipeline node {spec['name']!r} declares 'accept_if_contains' "
                "but has no transition with matching 'source'"
            )
        nodes.append(
            PromptStageNode(
                name=spec["name"],
                prompt=spec["prompt"],
                accept_if_contains=accept,
                jump_target=jump_target,
            )
        )
    return nodes


def _parse_pipeline(
    raw_pipeline: Any, problems: list[str]
) -> tuple[tuple[PromptStageNode, ...], tuple[Transition, ...]] | None:
    """Parse an optional `pipeline` section into workflow-graph stages (issue #228).

    `None` when the section is absent — the caller then runs the standard
    single-agent pipeline. When present, every node needs a unique `name` and
    a `prompt`; a node that also declares `accept_if_contains` is a decision
    stage and must be the `source` of exactly one `transitions` entry — its
    `target` becomes that node's jump target when the reply doesn't match.
    Appends every problem found to `problems` rather than raising, so the
    caller can report them all together.
    """
    if raw_pipeline in (None, ""):
        return None
    if not isinstance(raw_pipeline, dict):
        problems.append("'pipeline' section must be a mapping")
        return None
    unknown = set(raw_pipeline) - _PIPELINE_KEYS
    if unknown:
        problems.append(f"unknown 'pipeline' key(s): {', '.join(sorted(unknown))}")

    raw_nodes = raw_pipeline.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        problems.append("'pipeline.nodes' must be a non-empty list")
        return None

    node_names, node_specs = _parse_pipeline_node_specs(raw_nodes, problems)
    transitions, jump_target_by_source = _parse_pipeline_transitions(
        raw_pipeline.get("transitions"), node_names, problems
    )
    nodes = _build_pipeline_nodes(node_specs, jump_target_by_source, problems)

    return tuple(nodes), tuple(transitions)


def _parse_max_steps(run_section: dict[str, Any], problems: list[str]) -> int:
    max_steps_raw = run_section.get("max_steps")
    if not max_steps_raw:
        return DEFAULT_MAX_STEPS
    try:
        return int(max_steps_raw)
    except (TypeError, ValueError):
        problems.append(f"'run.max_steps' must be an integer, got {max_steps_raw!r}")
        return DEFAULT_MAX_STEPS


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
    if not checks:
        problems.append("'evaluation.checks' must declare at least one check")
    max_steps = _parse_max_steps(run_section, problems)
    pipeline_result = _parse_pipeline(data.get("pipeline"), problems)

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

    pipeline_nodes, pipeline_transitions = pipeline_result or (None, None)
    return LoadedRunConfig(
        run_config=run_config,
        checks=checks,
        content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        llm_section=llm_section,
        pipeline_nodes=pipeline_nodes,
        pipeline_transitions=pipeline_transitions,
    )


def _make_pipeline_builder(
    nodes: tuple[PromptStageNode, ...], transitions: tuple[Transition, ...]
) -> PipelineBuilder:
    """Return an `AgentNode.pipeline_builder` closing over a declared workflow graph.

    Matches `build_default_pipeline`'s call signature (`spawn_fn=`,
    `checkpoint_fn=`) — a config-declared graph has no context-reset spawning
    (no `ContextGuardNode`), so `spawn_fn` is accepted and ignored.
    """

    def _builder(spawn_fn: Any = None, checkpoint_fn: Any = None) -> Pipeline:  # noqa: ARG001
        return Pipeline(
            list(nodes), checkpoint_fn=checkpoint_fn, transitions=list(transitions)
        )

    return _builder


def _run_agent(
    run_config: RunConfig,
    label: str,
    llm_client: LLM,
    pipeline_builder: PipelineBuilder | None = None,
) -> tuple[str, RunResult]:
    """Run the agent once, in the real cwd, using ``llm_client``'s connection.

    Mirrors `evaluation._run_agent`, but also forwards `run_config`'s
    `max_steps` (carried in `extra_params`) to `AgentNode.execute`, which the
    reused helper does not need since evaluations don't configure it.

    `pipeline_builder` overrides the standard single-agent pipeline with a
    config-declared workflow graph (issue #228) — `None` (the default) runs
    every existing config exactly as before.
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
        pipeline_builder=pipeline_builder,
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
        scores.append(
            replace(scorer.score(case, run_result), session_id=run_result.session_id)
        )
    return scores


def execute_from_config(
    path: Path, *, results_root: Path | None = None
) -> tuple[EvalRunResult, str]:
    """Load, validate, and execute ``path`` end to end.

    Runs the config's `task` as a single real-cwd agent turn and scores it
    against every inline `evaluation.checks` entry, writing one result record
    stamped with the config's path and content hash.

    Returns:
        The written result record and its verdict (`"pass"`/`"fail"`).

    Raises:
        ConfigValidationError: If the config fails validation; no agent run
            or session directory is created.
    """
    loaded = load_config_file(path)

    llm_client = build_llm_client(loaded.llm_section)
    pipeline_builder = (
        _make_pipeline_builder(loaded.pipeline_nodes, loaded.pipeline_transitions or ())
        if loaded.pipeline_nodes
        else None
    )
    task, run_result = _run_agent(
        loaded.run_config,
        f"ConfigRun[{path.name}]",
        llm_client,
        pipeline_builder=pipeline_builder,
    )
    scores = _score_checks(loaded.checks, task, run_result)

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

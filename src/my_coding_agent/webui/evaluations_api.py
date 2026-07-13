"""Evaluation/RunConfig/EvalConfig CRUD + run HTTP API for the web UI shell.

A thin JSON-in/JSON-out layer over `evals.evaluation` — it never invents new
evaluation semantics, it only exposes CRUD and a "run" trigger over what
already exists there, dispatched under
`/api/evals/{evaluations,run-configs,eval-configs}`.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Protocol

from ..evals import evaluation as ev


class _Handler(Protocol):
    def _send_json(self, data: Any, status: int = 200) -> None: ...


def _read_json_body(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _error(handler: _Handler, message: str, *, status: int = 400) -> bool:
    handler._send_json({"error": message}, status=status)
    return True


# -- RunConfig ---------------------------------------------------------------


def _handle_run_configs_collection(
    handler: _Handler, method: str, raw_body: bytes, run_configs_dir: Path
) -> bool:
    if method == "GET":
        run_configs = ev.list_run_configs(base_dir=run_configs_dir)
        handler._send_json({"run_configs": [rc.to_dict() for rc in run_configs]})
        return True
    if method == "POST":
        body = _read_json_body(raw_body)
        if body is None or not isinstance(body.get("name"), str) or not body["name"]:
            return _error(handler, "name is required")
        run_config = ev.create_run_config(body, base_dir=run_configs_dir)
        handler._send_json(run_config.to_dict(), status=201)
        return True
    return False


def _handle_run_config_item(
    handler: _Handler,
    method: str,
    raw_body: bytes,
    run_configs_dir: Path,
    run_config_id: str,
) -> bool:
    if method == "GET":
        try:
            run_config = ev.get_run_config(run_config_id, base_dir=run_configs_dir)
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(run_config.to_dict())
        return True
    if method == "PUT":
        body = _read_json_body(raw_body)
        if body is None:
            return _error(handler, "invalid json")
        try:
            run_config = ev.update_run_config(
                run_config_id, body, base_dir=run_configs_dir
            )
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(run_config.to_dict())
        return True
    if method == "DELETE":
        try:
            ev.delete_run_config(run_config_id, base_dir=run_configs_dir)
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(None, status=204)
        return True
    return False


def _handle_run_configs(
    handler: _Handler, method: str, sub: str, raw_body: bytes, run_configs_dir: Path
) -> bool:
    if sub == "/run-configs":
        return _handle_run_configs_collection(
            handler, method, raw_body, run_configs_dir
        )
    match = re.fullmatch(r"/run-configs/([^/]+)", sub)
    if match:
        return _handle_run_config_item(
            handler, method, raw_body, run_configs_dir, match.group(1)
        )
    return False


# -- EvalConfig ----------------------------------------------------------


def _handle_eval_configs_collection(
    handler: _Handler, method: str, raw_body: bytes, eval_configs_dir: Path
) -> bool:
    if method == "GET":
        eval_configs = ev.list_eval_configs(base_dir=eval_configs_dir)
        handler._send_json({"eval_configs": [ec.to_dict() for ec in eval_configs]})
        return True
    if method == "POST":
        body = _read_json_body(raw_body)
        if body is None or not isinstance(body.get("name"), str) or not body["name"]:
            return _error(handler, "name is required")
        try:
            eval_config = ev.create_eval_config(body, base_dir=eval_configs_dir)
        except ev.UnknownEvaluatorError as exc:
            return _error(handler, str(exc))
        handler._send_json(eval_config.to_dict(), status=201)
        return True
    return False


def _handle_eval_config_item(
    handler: _Handler,
    method: str,
    raw_body: bytes,
    eval_configs_dir: Path,
    eval_config_id: str,
) -> bool:
    if method == "GET":
        try:
            eval_config = ev.get_eval_config(eval_config_id, base_dir=eval_configs_dir)
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(eval_config.to_dict())
        return True
    if method == "PUT":
        body = _read_json_body(raw_body)
        if body is None:
            return _error(handler, "invalid json")
        try:
            eval_config = ev.update_eval_config(
                eval_config_id, body, base_dir=eval_configs_dir
            )
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        except ev.UnknownEvaluatorError as exc:
            return _error(handler, str(exc))
        handler._send_json(eval_config.to_dict())
        return True
    if method == "DELETE":
        try:
            ev.delete_eval_config(eval_config_id, base_dir=eval_configs_dir)
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(None, status=204)
        return True
    return False


def _handle_eval_configs(
    handler: _Handler, method: str, sub: str, raw_body: bytes, eval_configs_dir: Path
) -> bool:
    if sub == "/eval-configs":
        return _handle_eval_configs_collection(
            handler, method, raw_body, eval_configs_dir
        )
    match = re.fullmatch(r"/eval-configs/([^/]+)", sub)
    if match:
        return _handle_eval_config_item(
            handler, method, raw_body, eval_configs_dir, match.group(1)
        )
    return False


# -- Evaluation ------------------------------------------------------------


def _handle_evaluations_collection(
    handler: _Handler,
    method: str,
    raw_body: bytes,
    *,
    evaluations_dir: Path,
    run_configs_dir: Path,
    eval_configs_dir: Path,
) -> bool:
    if method == "GET":
        evaluations = ev.list_evaluations(base_dir=evaluations_dir)
        handler._send_json({"evaluations": [e.to_dict() for e in evaluations]})
        return True
    if method == "POST":
        body = _read_json_body(raw_body)
        if (
            body is None
            or not isinstance(body.get("name"), str)
            or not body["name"]
            or not isinstance(body.get("run_config_id"), str)
            or not isinstance(body.get("eval_config_id"), str)
        ):
            return _error(handler, "name, run_config_id, eval_config_id are required")
        try:
            evaluation = ev.create_evaluation(
                body,
                base_dir=evaluations_dir,
                run_configs_dir=run_configs_dir,
                eval_configs_dir=eval_configs_dir,
            )
        except ev.UnresolvedReferenceError as exc:
            return _error(handler, str(exc))
        handler._send_json(evaluation.to_dict(), status=201)
        return True
    return False


def _handle_evaluation_item(
    handler: _Handler,
    method: str,
    raw_body: bytes,
    *,
    evaluations_dir: Path,
    run_configs_dir: Path,
    eval_configs_dir: Path,
    evaluation_id: str,
) -> bool:
    if method == "GET":
        try:
            evaluation = ev.get_evaluation(evaluation_id, base_dir=evaluations_dir)
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(evaluation.to_dict())
        return True
    if method == "PUT":
        body = _read_json_body(raw_body)
        if body is None:
            return _error(handler, "invalid json")
        try:
            evaluation = ev.update_evaluation(
                evaluation_id,
                body,
                base_dir=evaluations_dir,
                run_configs_dir=run_configs_dir,
                eval_configs_dir=eval_configs_dir,
            )
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        except ev.UnresolvedReferenceError as exc:
            return _error(handler, str(exc))
        handler._send_json(evaluation.to_dict())
        return True
    if method == "DELETE":
        try:
            ev.delete_evaluation(evaluation_id, base_dir=evaluations_dir)
        except ev.NotFoundError as exc:
            return _error(handler, str(exc), status=404)
        handler._send_json(None, status=204)
        return True
    return False


def _handle_evaluation_run(
    handler: _Handler,
    method: str,
    *,
    evaluations_dir: Path,
    run_configs_dir: Path,
    eval_configs_dir: Path,
    evaluation_id: str,
) -> bool:
    if method != "POST":
        return False
    try:
        evaluation = ev.get_evaluation(evaluation_id, base_dir=evaluations_dir)
    except ev.NotFoundError as exc:
        return _error(handler, str(exc), status=404)
    run_id = ev.new_id()

    def _run() -> None:
        try:
            ev.run_evaluation(
                evaluation,
                run_configs_dir=run_configs_dir,
                eval_configs_dir=eval_configs_dir,
                evaluations_dir=evaluations_dir,
                results_root=evaluations_dir.parent,
                run_id=run_id,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "evaluation %s: background run %s failed", evaluation_id, run_id
            )

    # 202 means "accepted, not yet done" — the run happens on a background
    # thread (server.py's ThreadingHTTPServer keeps the rest of the UI, and
    # this poll endpoint, responsive while it runs) rather than blocking this
    # request until the agent turn completes.
    threading.Thread(target=_run, daemon=True).start()
    handler._send_json({"run_id": run_id}, status=202)
    return True


def _handle_evaluation_run_result(
    handler: _Handler, method: str, *, evaluations_dir: Path, run_id: str
) -> bool:
    if method != "GET":
        return False
    result_path = evaluations_dir.parent / run_id / "result.json"
    if not result_path.exists():
        handler._send_json({"run_id": run_id, "status": "running"}, status=202)
        return True
    handler._send_json(json.loads(result_path.read_text()))
    return True


def _handle_evaluations(
    handler: _Handler,
    method: str,
    sub: str,
    raw_body: bytes,
    *,
    evaluations_dir: Path,
    run_configs_dir: Path,
    eval_configs_dir: Path,
) -> bool:
    if sub == "/evaluations":
        return _handle_evaluations_collection(
            handler,
            method,
            raw_body,
            evaluations_dir=evaluations_dir,
            run_configs_dir=run_configs_dir,
            eval_configs_dir=eval_configs_dir,
        )

    match = re.fullmatch(r"/evaluations/([^/]+)/run", sub)
    if match:
        return _handle_evaluation_run(
            handler,
            method,
            evaluations_dir=evaluations_dir,
            run_configs_dir=run_configs_dir,
            eval_configs_dir=eval_configs_dir,
            evaluation_id=match.group(1),
        )

    match = re.fullmatch(r"/evaluations/([^/]+)/runs/([^/]+)", sub)
    if match:
        return _handle_evaluation_run_result(
            handler, method, evaluations_dir=evaluations_dir, run_id=match.group(2)
        )

    match = re.fullmatch(r"/evaluations/([^/]+)", sub)
    if match:
        return _handle_evaluation_item(
            handler,
            method,
            raw_body,
            evaluations_dir=evaluations_dir,
            run_configs_dir=run_configs_dir,
            eval_configs_dir=eval_configs_dir,
            evaluation_id=match.group(1),
        )

    return False


def handle_evaluation_route(
    handler: _Handler,
    method: str,
    path: str,
    raw_body: bytes,
    *,
    evals_root: Path,
) -> bool:
    """Dispatch a `/api/evals/{evaluations,run-configs,eval-configs}` request.

    Routes:
        GET    /evaluations                    list evaluations
        POST   /evaluations                    create an Evaluation
        GET    /evaluations/{id}               read
        PUT    /evaluations/{id}               update
        DELETE /evaluations/{id}                delete (never cascades)
        POST   /evaluations/{id}/run           run -> 202 {run_id}
        GET    /evaluations/{id}/runs/{run_id} read a run's result record
        GET    /run-configs                    list run configs
        POST   /run-configs                    create (RunConfig, no id)
        GET|PUT|DELETE /run-configs/{id}       read / update / delete
        GET    /eval-configs                   list eval configs
        POST   /eval-configs                   create (EvalConfig, no id)
        GET|PUT|DELETE /eval-configs/{id}      read / update / delete
    """
    if not path.startswith("/api/evals/"):
        return False
    sub = path[len("/api/evals") :]
    if sub.startswith("/config"):
        return False

    evaluations_dir = evals_root / "evaluations"
    run_configs_dir = evals_root / "run_configs"
    eval_configs_dir = evals_root / "eval_configs"

    return (
        _handle_run_configs(handler, method, sub, raw_body, run_configs_dir)
        or _handle_eval_configs(handler, method, sub, raw_body, eval_configs_dir)
        or _handle_evaluations(
            handler,
            method,
            sub,
            raw_body,
            evaluations_dir=evaluations_dir,
            run_configs_dir=run_configs_dir,
            eval_configs_dir=eval_configs_dir,
        )
    )

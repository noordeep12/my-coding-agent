"""Eval configuration CRUD API and the run-to-eval bridge, served from the
unified web UI shell.

A thin JSON-in/JSON-out layer over the existing `evals` model
(`evals.datasets`, `evals.cases`, `evals.scoring`) — it never invents new
evaluation semantics, it only exposes CRUD and a "run" trigger over what
already exists. Extracts a completed session's task/output directly from its
`events.jsonl` for the run-to-eval bridge (`evals.add_failure_case`).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Protocol

from ..evals import cases as cases_mod
from ..evals import datasets as eval_datasets
from ..evals import scoring as scoring_mod
from ..evals.schema import EvalCase

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SID_RE = re.compile(r"^[0-9a-f]{8,64}$")

_DRAFT_TABLE = "eval_config_draft"


class _Handler(Protocol):
    def _send_json(self, data: Any, status: int = 200) -> None: ...


class Store(Protocol):
    def get_item(self, table_name: str, item_id: str) -> dict[str, Any] | None: ...
    def create_item(
        self, table_name: str, item_id: str, payload: dict[str, Any]
    ) -> None: ...
    def update_item(
        self, table_name: str, item_id: str, payload: dict[str, Any]
    ) -> None: ...


def _case_to_dict(case: EvalCase) -> dict[str, Any]:
    return dataclasses.asdict(case)


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    events = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _extract_task(events: list[dict[str, Any]]) -> str | None:
    for ev in events:
        messages = ev.get("messages")
        if not isinstance(messages, list):
            continue
        user = next(
            (m.get("content") for m in messages if m.get("role") == "user"), None
        )
        if isinstance(user, str):
            return user
    start_ev = next((ev for ev in events if ev.get("type") == "session_start"), None)
    return (start_ev or {}).get("label")


def _extract_final_output(events: list[dict[str, Any]]) -> str | None:
    for ev in reversed(events):
        messages = ev.get("messages")
        if not isinstance(messages, list):
            continue
        for m in reversed(messages):
            content = m.get("content")
            if m.get("role") == "assistant" and isinstance(content, str):
                return content
    return None


def _session_task_and_output(session_dir: Path) -> tuple[str | None, str | None]:
    """Best-effort extraction of a session's task prompt and final output.

    Reads `events.jsonl` directly (same event shape `evals.runner` produces
    for a live case run) rather than the full trace graph, since only the
    opening user message and the last assistant message are needed here.
    """
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return None, None
    events = _read_events(events_path)
    return _extract_task(events), _extract_final_output(events)


def _read_json_body(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _handle_dataset_create(
    handler: _Handler, raw_body: bytes, datasets_dir: Path
) -> bool:
    body = _read_json_body(raw_body)
    if body is None or not isinstance(body.get("id"), str) or not body["id"]:
        handler._send_json({"error": "id is required"}, status=400)
        return True
    case_ids = body.get("case_ids") or []
    if not isinstance(case_ids, list) or not all(isinstance(c, str) for c in case_ids):
        handler._send_json({"error": "case_ids must be strings"}, status=400)
        return True
    try:
        dataset = eval_datasets.create_dataset(
            body["id"], case_ids, base_dir=datasets_dir
        )
    except eval_datasets.DatasetError as exc:
        handler._send_json({"error": str(exc)}, status=400)
        return True
    handler._send_json(dataclasses.asdict(dataset))
    return True


def _handle_dataset_add_case(
    handler: _Handler, raw_body: bytes, datasets_dir: Path, dataset_id: str
) -> bool:
    body = _read_json_body(raw_body)
    case_id = body.get("case_id") if body else None
    if not isinstance(case_id, str) or not case_id:
        handler._send_json({"error": "case_id is required"}, status=400)
        return True
    try:
        dataset = eval_datasets.add_case(
            dataset_id,
            case_id,
            base_dir=datasets_dir,
            note=body.get("note", "") if body else "",
        )
    except eval_datasets.DatasetError as exc:
        handler._send_json({"error": str(exc)}, status=400)
        return True
    handler._send_json(dataclasses.asdict(dataset))
    return True


def _handle_datasets(
    handler: _Handler, method: str, sub: str, raw_body: bytes, datasets_dir: Path
) -> bool:
    if sub == "/datasets" and method == "GET":
        handler._send_json(
            [
                dataclasses.asdict(d)
                for d in eval_datasets.list_datasets(base_dir=datasets_dir)
            ]
        )
        return True

    if sub == "/datasets" and method == "POST":
        return _handle_dataset_create(handler, raw_body, datasets_dir)

    match = re.fullmatch(r"/datasets/([^/]+)/cases", sub)
    if match and method == "POST":
        return _handle_dataset_add_case(handler, raw_body, datasets_dir, match.group(1))

    match = re.fullmatch(r"/datasets/([^/]+)/cases/([^/]+)", sub)
    if match and method == "DELETE":
        dataset_id, case_id = match.group(1), match.group(2)
        try:
            dataset = eval_datasets.retire_case(
                dataset_id, case_id, base_dir=datasets_dir
            )
        except eval_datasets.DatasetError as exc:
            handler._send_json({"error": str(exc)}, status=400)
            return True
        handler._send_json(dataclasses.asdict(dataset))
        return True

    return False


def _handle_cases(
    handler: _Handler, method: str, sub: str, raw_body: bytes, cases_dir: Path
) -> bool:
    if sub == "/cases" and method == "GET":
        handler._send_json(
            [_case_to_dict(c) for c in cases_mod.load_case_set(cases_dir)]
        )
        return True

    if sub == "/cases" and method == "POST":
        return _handle_case_create(handler, raw_body, cases_dir)

    match = re.fullmatch(r"/cases/([^/]+)", sub)
    if match and method == "DELETE":
        try:
            cases_mod.delete_case(match.group(1), cases_dir)
        except cases_mod.CaseNotFoundError as exc:
            handler._send_json({"error": str(exc)}, status=404)
            return True
        handler._send_json({"ok": True})
        return True

    if sub == "/scorers" and method == "GET":
        handler._send_json(scoring_mod.list_scorer_refs())
        return True

    return False


def _handle_case_create(handler: _Handler, raw_body: bytes, cases_dir: Path) -> bool:
    body = _read_json_body(raw_body)
    if body is None:
        handler._send_json({"error": "invalid json"}, status=400)
        return True
    case_id, task, scorer, expected = (
        body.get("id"),
        body.get("task"),
        body.get("scorer"),
        body.get("expected"),
    )
    if (
        not isinstance(case_id, str)
        or not _CASE_ID_RE.match(case_id)
        or not isinstance(task, str)
        or not task
        or not isinstance(scorer, str)
        or not isinstance(expected, dict)
    ):
        handler._send_json(
            {"error": "id, task, scorer, expected (object) are required"}, status=400
        )
        return True
    problems = scoring_mod.validate_expected(scorer, expected)
    if problems:
        handler._send_json({"error": "; ".join(problems)}, status=400)
        return True
    case = EvalCase(
        id=case_id,
        task=task,
        scorer=scorer,
        expected=expected,
        dataset=body.get("dataset"),
        tags=tuple(body.get("tags") or []),
    )
    cases_mod.save_case(case, cases_dir)
    handler._send_json(_case_to_dict(case))
    return True


def _handle_run(
    handler: _Handler, method: str, sub: str, raw_body: bytes, evals_root: Path
) -> bool:
    if sub != "/run" or method != "POST":
        return False
    datasets_dir = evals_root / "datasets"
    cases_dir = evals_root / "cases"
    body = _read_json_body(raw_body)
    dataset_id = body.get("dataset_id") if body else None
    if not isinstance(dataset_id, str) or not dataset_id:
        handler._send_json({"error": "dataset_id is required"}, status=400)
        return True
    try:
        # nosec B615 — this is evals.datasets.load_dataset (on-disk versioned
        # case-id list), not huggingface_hub's dataset downloader; bandit's
        # rule matches on the function name alone.
        dataset = eval_datasets.load_dataset(dataset_id, base_dir=datasets_dir)  # nosec B615
    except eval_datasets.DatasetNotFoundError as exc:
        handler._send_json({"error": str(exc)}, status=404)
        return True
    result = eval_datasets.run_dataset(
        dataset, cases_dir=cases_dir, results_root=evals_root
    )
    handler._send_json(dataclasses.asdict(result))
    return True


def _handle_send_run(
    handler: _Handler,
    method: str,
    sub: str,
    raw_body: bytes,
    evals_root: Path,
    sessions_root: Path,
) -> bool:
    if sub != "/send-run" or method != "POST":
        return False
    datasets_dir = evals_root / "datasets"
    cases_dir = evals_root / "cases"
    body = _read_json_body(raw_body)
    session_id = body.get("session_id") if body else None
    dataset_id = body.get("dataset_id") if body else None
    scorer = (body.get("scorer") if body else None) or "exact_match"
    if (
        not isinstance(session_id, str)
        or not _SID_RE.match(session_id)
        or not isinstance(dataset_id, str)
        or not dataset_id
    ):
        handler._send_json(
            {"error": "session_id and dataset_id are required"}, status=400
        )
        return True
    session_dir = (sessions_root.resolve() / session_id).resolve()
    if not session_dir.is_relative_to(sessions_root.resolve()):
        handler._send_json({"error": "invalid session id"}, status=400)
        return True
    task, output = _session_task_and_output(session_dir)
    if task is None:
        handler._send_json(
            {"error": f"session '{session_id}' has no readable task"}, status=404
        )
        return True
    case_id = (body or {}).get("case_id") or f"run-{session_id[:12]}"
    expected = (body or {}).get("expected") or {"equals": output or ""}
    try:
        dataset = eval_datasets.add_failure_case(
            dataset_id,
            case_id=case_id,
            task=task,
            scorer=scorer,
            expected=expected,
            base_dir=datasets_dir,
            cases_dir=cases_dir,
        )
    except eval_datasets.DatasetError as exc:
        handler._send_json({"error": str(exc)}, status=400)
        return True
    handler._send_json(dataclasses.asdict(dataset))
    return True


def _handle_draft(
    handler: _Handler, method: str, sub: str, raw_body: bytes, store: Store
) -> bool:
    if sub == "/draft" and method == "GET":
        handler._send_json(store.get_item(_DRAFT_TABLE, "current") or {})
        return True

    if sub == "/draft" and method == "POST":
        body = _read_json_body(raw_body)
        if body is None:
            handler._send_json({"error": "invalid json"}, status=400)
            return True
        if store.get_item(_DRAFT_TABLE, "current") is None:
            store.create_item(_DRAFT_TABLE, "current", body)
        else:
            store.update_item(_DRAFT_TABLE, "current", body)
        handler._send_json({"ok": True})
        return True

    return False


def handle_eval_config_route(
    handler: _Handler,
    method: str,
    path: str,
    raw_body: bytes,
    *,
    evals_root: Path,
    sessions_root: Path,
    store: Store,
) -> bool:
    """Dispatch a `/api/evals/config/...` request; return True if handled.

    Routes (all under `/api/evals/config`):
        GET    /datasets                      list datasets
        POST   /datasets                      create dataset {id, case_ids}
        POST   /datasets/{id}/cases           add case {case_id, note}
        DELETE /datasets/{id}/cases/{case_id} retire case
        GET    /cases                         list cases
        POST   /cases                         create/update case
        DELETE /cases/{id}                    delete case
        GET    /scorers                       list registered scorer refs
        POST   /run                           run a dataset {dataset_id}
        POST   /send-run                      run-to-eval bridge
        GET    /draft                         UI-only draft/selection state
        POST   /draft                         persist UI-only draft state
    """
    if not path.startswith("/api/evals/config"):
        return False
    sub = path[len("/api/evals/config") :] or "/"
    datasets_dir = evals_root / "datasets"
    cases_dir = evals_root / "cases"

    return (
        _handle_datasets(handler, method, sub, raw_body, datasets_dir)
        or _handle_cases(handler, method, sub, raw_body, cases_dir)
        or _handle_run(handler, method, sub, raw_body, evals_root)
        or _handle_send_run(handler, method, sub, raw_body, evals_root, sessions_root)
        or _handle_draft(handler, method, sub, raw_body, store)
    )

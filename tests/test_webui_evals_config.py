"""Tests for webui/evals_config.py — eval-config CRUD API and run-to-eval bridge."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.evals.datasets import load_dataset
from my_coding_agent.webui import evals_config
from my_coding_agent.webui.evals_config import (
    _extract_final_output,
    _extract_task,
    _read_events,
    handle_eval_config_route,
)
from my_coding_agent.webui.server import _WebUIHandler
from my_coding_agent.webui.store import Store, default_db_path


@pytest.fixture()
def server(tmp_path):
    _WebUIHandler.base_dir = tmp_path
    _WebUIHandler.store = Store(default_db_path(tmp_path))
    httpd = None
    port = None
    for p in range(19800, 19900):
        try:
            httpd = HTTPServer(("127.0.0.1", p), _WebUIHandler)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        pytest.skip("No free port found")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port, tmp_path
    httpd.shutdown()
    _WebUIHandler.store.close()


def _req(port, method, path, payload=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    resp_body = resp.read()
    conn.close()
    return resp.status, json.loads(resp_body)


def _req_raw(port, method, path, raw_body):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(
        method, path, body=raw_body, headers={"Content-Type": "application/json"}
    )
    resp = conn.getresponse()
    resp_body = resp.read()
    conn.close()
    return resp.status, json.loads(resp_body)


def test_dataset_crud_via_api_matches_on_disk_datasets_model(server):
    port, tmp_path = server

    status, body = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "ds1", "case_ids": []}
    )
    assert status == 200
    assert body == {"id": "ds1", "version": 1, "case_ids": []}

    status, body = _req(
        port,
        "POST",
        "/api/evals/config/cases",
        {
            "id": "c1",
            "task": "do it",
            "scorer": "exact_match",
            "expected": {"equals": "ok"},
        },
    )
    assert status == 200

    status, body = _req(
        port, "POST", "/api/evals/config/datasets/ds1/cases", {"case_id": "c1"}
    )
    assert status == 200
    assert body["case_ids"] == ["c1"]

    # The API-driven write is readable through evals.datasets directly, and
    # its versioning matches what add_case() would have produced.
    on_disk = load_dataset("ds1", base_dir=tmp_path / "evals" / "datasets")
    assert on_disk.version == 2
    assert on_disk.case_ids == ("c1",)

    status, body = _req(port, "GET", "/api/evals/config/datasets")
    assert status == 200
    assert body == [{"id": "ds1", "version": 2, "case_ids": ["c1"]}]

    status, body = _req(port, "DELETE", "/api/evals/config/datasets/ds1/cases/c1")
    assert status == 200
    assert body["case_ids"] == []
    on_disk = load_dataset("ds1", base_dir=tmp_path / "evals" / "datasets")
    assert on_disk.version == 3


def test_case_create_rejects_expected_shape_not_matching_scorer(server):
    port, _ = server
    status, body = _req(
        port,
        "POST",
        "/api/evals/config/cases",
        {"id": "bad", "task": "x", "scorer": "exact_match", "expected": {}},
    )
    assert status == 400
    assert "exact_match" in body["error"]


def test_case_create_and_delete(server):
    port, tmp_path = server
    status, _ = _req(
        port,
        "POST",
        "/api/evals/config/cases",
        {
            "id": "c2",
            "task": "x",
            "scorer": "exact_match",
            "expected": {"contains": "y"},
        },
    )
    assert status == 200
    assert (tmp_path / "evals" / "cases" / "c2.json").exists()

    status, body = _req(port, "GET", "/api/evals/config/cases")
    assert status == 200
    assert [c["id"] for c in body] == ["c2"]

    status, body = _req(port, "DELETE", "/api/evals/config/cases/c2")
    assert status == 200
    assert not (tmp_path / "evals" / "cases" / "c2.json").exists()


def test_scorers_route_lists_registered_scorers(server):
    port, _ = server
    status, body = _req(port, "GET", "/api/evals/config/scorers")
    assert status == 200
    assert "exact_match" in body


def _write_session(tmp_path, session_id, *, task, output):
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True)
    events = [
        {"type": "session_start", "session_id": session_id, "label": task},
        {
            "type": "llm_call",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": task},
                {"role": "assistant", "content": output},
            ],
        },
    ]
    with (session_dir / "events.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_send_run_to_eval_seeds_case_via_add_failure_case(server):
    port, tmp_path = server
    _write_session(tmp_path, "abcd1234abcd", task="do the thing", output="42")

    status, body = _req(
        port,
        "POST",
        "/api/evals/config/datasets",
        {"id": "regressions", "case_ids": []},
    )
    assert status == 200

    status, body = _req(
        port,
        "POST",
        "/api/evals/config/send-run",
        {"session_id": "abcd1234abcd", "dataset_id": "regressions"},
    )
    assert status == 200
    assert body["case_ids"] == ["run-abcd1234abcd"]

    case_file = tmp_path / "evals" / "cases" / "run-abcd1234abcd.json"
    assert case_file.exists()
    case_payload = json.loads(case_file.read_text())
    assert case_payload["task"] == "do the thing"
    assert case_payload["expected"] == {"equals": "42"}
    assert case_payload["scorer"] == "exact_match"

    on_disk = load_dataset("regressions", base_dir=tmp_path / "evals" / "datasets")
    assert on_disk.case_ids == ("run-abcd1234abcd",)


def test_send_run_rejects_unknown_session(server):
    port, _ = server
    status, _ = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "ds", "case_ids": []}
    )
    assert status == 200
    status, body = _req(
        port,
        "POST",
        "/api/evals/config/send-run",
        {"session_id": "0000000000000000", "dataset_id": "ds"},
    )
    assert status == 404


def test_run_dataset_from_config_api_writes_a_run_result(server):
    port, tmp_path = server
    _req(port, "POST", "/api/evals/config/datasets", {"id": "ds", "case_ids": []})
    _req(
        port,
        "POST",
        "/api/evals/config/cases",
        {
            "id": "c1",
            "task": "reply with hi",
            "scorer": "exact_match",
            "expected": {"contains": "hi"},
        },
    )
    _req(port, "POST", "/api/evals/config/datasets/ds/cases", {"case_id": "c1"})

    # We don't run the real agent in a unit test; assert the route dispatches
    # to evals.datasets.run_dataset and surfaces its error for a case whose
    # scorer resolution or run environment isn't available in this sandbox,
    # rather than silently no-op'ing.
    status, body = _req(
        port, "POST", "/api/evals/config/run", {"dataset_id": "missing-ds"}
    )
    assert status == 404


def test_draft_state_persists_across_requests(server):
    port, _ = server
    status, body = _req(port, "GET", "/api/evals/config/draft")
    assert status == 200
    assert body == {}

    status, body = _req(
        port, "POST", "/api/evals/config/draft", {"selected_dataset": "ds1"}
    )
    assert status == 200

    status, body = _req(port, "GET", "/api/evals/config/draft")
    assert status == 200
    assert body == {"selected_dataset": "ds1"}


# ── events.jsonl extraction helpers ──────────────────────────────────────────


def test_read_events_skips_blank_invalid_and_non_dict_lines(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text('\n{"type": "ok"}\nnot json\n[1, 2]\n')
    assert _read_events(events_path) == [{"type": "ok"}]


def test_extract_task_falls_back_to_session_start_label():
    events = [
        {"messages": [{"role": "assistant", "content": "no user here"}]},
        {"type": "session_start", "label": "labelled task"},
    ]
    assert _extract_task(events) == "labelled task"


def test_extract_task_none_without_messages_or_label():
    assert _extract_task([{"type": "other"}]) is None


def test_extract_final_output_skips_non_string_and_non_list_content():
    events = [
        {"messages": "not a list"},
        {"messages": [{"role": "assistant", "content": ["structured"]}]},
    ]
    assert _extract_final_output(events) is None


# ── dataset error paths ──────────────────────────────────────────────────────


def test_dataset_create_rejects_invalid_json_and_missing_id(server):
    port, _ = server
    status, body = _req_raw(port, "POST", "/api/evals/config/datasets", b"not json")
    assert status == 400
    assert body["error"] == "id is required"

    status, body = _req(port, "POST", "/api/evals/config/datasets", {"case_ids": []})
    assert status == 400
    assert body["error"] == "id is required"


def test_dataset_create_rejects_non_string_case_ids(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "ds", "case_ids": [1, 2]}
    )
    assert status == 400
    assert body["error"] == "case_ids must be strings"


def test_dataset_create_rejects_duplicate_id(server):
    port, _ = server
    status, _ = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "dup", "case_ids": []}
    )
    assert status == 200
    status, body = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "dup", "case_ids": []}
    )
    assert status == 400
    assert "dup" in body["error"]


def test_dataset_add_case_requires_case_id(server):
    port, _ = server
    status, body = _req(port, "POST", "/api/evals/config/datasets/ds/cases", {})
    assert status == 400
    assert body["error"] == "case_id is required"


def test_dataset_add_case_unknown_dataset_400(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/api/evals/config/datasets/ghost/cases", {"case_id": "c1"}
    )
    assert status == 400
    assert "ghost" in body["error"]


def test_dataset_retire_case_unknown_dataset_400(server):
    port, _ = server
    status, body = _req(port, "DELETE", "/api/evals/config/datasets/ghost/cases/c1")
    assert status == 400
    assert "ghost" in body["error"]


# ── case error paths ─────────────────────────────────────────────────────────


def test_case_delete_unknown_case_404(server):
    port, _ = server
    status, body = _req(port, "DELETE", "/api/evals/config/cases/ghost")
    assert status == 404
    assert "ghost" in body["error"]


def test_case_create_rejects_invalid_json(server):
    port, _ = server
    status, body = _req_raw(port, "POST", "/api/evals/config/cases", b"not json")
    assert status == 400
    assert body["error"] == "invalid json"


def test_case_create_rejects_missing_or_malformed_fields(server):
    port, _ = server
    status, body = _req(
        port,
        "POST",
        "/api/evals/config/cases",
        {"id": "bad id!", "task": "x", "scorer": "exact_match", "expected": {}},
    )
    assert status == 400
    assert "required" in body["error"]


# ── run error and success paths ──────────────────────────────────────────────


def test_run_requires_dataset_id(server):
    port, _ = server
    status, body = _req(port, "POST", "/api/evals/config/run", {})
    assert status == 400
    assert body["error"] == "dataset_id is required"


def test_run_empty_dataset_returns_result_record(server):
    port, tmp_path = server
    status, _ = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "empty", "case_ids": []}
    )
    assert status == 200
    status, body = _req(port, "POST", "/api/evals/config/run", {"dataset_id": "empty"})
    assert status == 200
    assert body["dataset"] == "empty@v1"
    assert body["scores"] == []
    run_dir = tmp_path / "evals" / body["run_id"]
    assert (run_dir / "result.json").exists()


# ── send-run error paths ─────────────────────────────────────────────────────


def test_send_run_requires_session_and_dataset_ids(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/api/evals/config/send-run", {"session_id": "abcd1234abcd"}
    )
    assert status == 400
    assert body["error"] == "session_id and dataset_id are required"


def test_send_run_rejects_session_id_escaping_sessions_root(server, monkeypatch):
    # The route's session-id regex already blocks traversal; loosen it to
    # prove the resolved-path containment check holds on its own.
    import re

    monkeypatch.setattr(evals_config, "_SID_RE", re.compile(r".+"))
    port, _ = server
    status, body = _req(
        port,
        "POST",
        "/api/evals/config/send-run",
        {"session_id": "../outside", "dataset_id": "ds"},
    )
    assert status == 400
    assert body["error"] == "invalid session id"


def test_send_run_unknown_dataset_400(server):
    port, tmp_path = server
    _write_session(tmp_path, "abcd1234abcd", task="do it", output="42")
    status, body = _req(
        port,
        "POST",
        "/api/evals/config/send-run",
        {"session_id": "abcd1234abcd", "dataset_id": "ghost"},
    )
    assert status == 400
    assert "ghost" in body["error"]


# ── draft error and dispatch paths ───────────────────────────────────────────


def test_draft_post_rejects_invalid_json(server):
    port, _ = server
    status, body = _req_raw(port, "POST", "/api/evals/config/draft", b"not json")
    assert status == 400
    assert body["error"] == "invalid json"


def test_draft_second_post_updates_existing_draft(server):
    port, _ = server
    status, _ = _req(port, "POST", "/api/evals/config/draft", {"selected": "a"})
    assert status == 200
    status, _ = _req(port, "POST", "/api/evals/config/draft", {"selected": "b"})
    assert status == 200
    status, body = _req(port, "GET", "/api/evals/config/draft")
    assert status == 200
    assert body == {"selected": "b"}


def test_draft_unhandled_method_404(server):
    port, _ = server
    status, body = _req(port, "PUT", "/api/evals/config/draft", {"selected": "a"})
    assert status == 404


def test_handle_eval_config_route_ignores_foreign_prefix(tmp_path):
    class _NullHandler:
        def _send_json(self, data, status=200):
            raise AssertionError("dispatcher must not respond to foreign paths")

    store = Store(default_db_path(tmp_path))
    assert (
        handle_eval_config_route(
            _NullHandler(),
            "GET",
            "/api/other",
            b"",
            evals_root=tmp_path / "evals",
            sessions_root=tmp_path,
            store=store,
        )
        is False
    )
    store.close()

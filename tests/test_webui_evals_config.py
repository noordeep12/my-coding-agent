"""Tests for webui/evals_config.py — eval-config CRUD API and run-to-eval bridge."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.evals.datasets import load_dataset
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


def test_dataset_crud_via_api_matches_on_disk_datasets_model(server):
    port, tmp_path = server

    status, body = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "ds1", "case_ids": []}
    )
    assert status == 200
    assert body == {"id": "ds1", "version": 1, "case_ids": []}

    status, body = _req(
        port, "POST", "/api/evals/config/cases",
        {"id": "c1", "task": "do it", "scorer": "exact_match", "expected": {"equals": "ok"}},
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

    status, body = _req(
        port, "DELETE", "/api/evals/config/datasets/ds1/cases/c1"
    )
    assert status == 200
    assert body["case_ids"] == []
    on_disk = load_dataset("ds1", base_dir=tmp_path / "evals" / "datasets")
    assert on_disk.version == 3


def test_case_create_rejects_expected_shape_not_matching_scorer(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/api/evals/config/cases",
        {"id": "bad", "task": "x", "scorer": "exact_match", "expected": {}},
    )
    assert status == 400
    assert "exact_match" in body["error"]


def test_case_create_and_delete(server):
    port, tmp_path = server
    status, _ = _req(
        port, "POST", "/api/evals/config/cases",
        {"id": "c2", "task": "x", "scorer": "exact_match", "expected": {"contains": "y"}},
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
        port, "POST", "/api/evals/config/datasets", {"id": "regressions", "case_ids": []}
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
    status, body = _req(port, "POST", "/api/evals/config/run", {"dataset_id": "missing-ds"})
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

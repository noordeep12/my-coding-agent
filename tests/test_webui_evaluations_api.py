"""Tests for webui/evaluations_api.py — Evaluation/RunConfig/EvalConfig HTTP API."""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.engine.agent import AgentNode
from my_coding_agent.webui.server import _WebUIHandler
from my_coding_agent.webui.store import Store, default_db_path


@pytest.fixture()
def server(tmp_path):
    _WebUIHandler.base_dir = tmp_path
    _WebUIHandler.store = Store(default_db_path(tmp_path))
    httpd = None
    port = None
    for p in range(19900, 20000):
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
    parsed = json.loads(resp_body) if resp_body else None
    return resp.status, parsed


def test_run_config_crud(server):
    port, _ = server

    status, body = _req(port, "POST", "/api/evals/run-configs", {"name": "rc1"})
    assert status == 201
    rc_id = body["id"]

    status, body = _req(port, "GET", "/api/evals/run-configs")
    assert status == 200
    assert [rc["id"] for rc in body["run_configs"]] == [rc_id]

    status, body = _req(port, "GET", f"/api/evals/run-configs/{rc_id}")
    assert status == 200
    assert body["name"] == "rc1"

    status, body = _req(
        port, "PUT", f"/api/evals/run-configs/{rc_id}", {"name": "rc1-renamed"}
    )
    assert status == 200
    assert body["name"] == "rc1-renamed"

    status, body = _req(port, "DELETE", f"/api/evals/run-configs/{rc_id}")
    assert status == 204

    status, body = _req(port, "GET", f"/api/evals/run-configs/{rc_id}")
    assert status == 404


def test_eval_config_crud_with_rules_and_checks(server):
    port, _ = server

    status, body = _req(
        port,
        "POST",
        "/api/evals/eval-configs",
        {
            "name": "ec1",
            "rules": [
                {
                    "name": "rule1",
                    "checks": [
                        {"name": "c1", "evaluator": "exact_match", "expected": "pong"},
                        {"name": "c2", "evaluator": "exact_match", "expected": "pong"},
                    ],
                }
            ],
        },
    )
    assert status == 201
    assert len(body["rules"]) == 1
    assert len(body["rules"][0]["checks"]) == 2
    ec_id = body["id"]

    status, body = _req(port, "GET", "/api/evals/eval-configs")
    assert status == 200
    assert [ec["id"] for ec in body["eval_configs"]] == [ec_id]


def test_eval_config_create_rejects_unknown_evaluator(server):
    port, _ = server
    status, body = _req(
        port,
        "POST",
        "/api/evals/eval-configs",
        {
            "name": "bad",
            "rules": [{"name": "r1", "checks": [{"name": "c1", "evaluator": "nope"}]}],
        },
    )
    assert 400 <= status < 500
    assert "nope" in body["error"] or "evaluator" in body["error"]


def test_evaluation_create_rejects_missing_config_reference(server):
    port, _ = server
    status, body = _req(port, "POST", "/api/evals/run-configs", {"name": "rc1"})
    rc_id = body["id"]

    status, body = _req(
        port,
        "POST",
        "/api/evals/evaluations",
        {"name": "e1", "run_config_id": rc_id, "eval_config_id": "missing"},
    )
    assert 400 <= status < 500
    assert "eval_config_id" in body["error"] or "missing" in body["error"]


def test_evaluation_full_lifecycle_run_and_result(server, monkeypatch, mocker):
    port, tmp_path = server
    monkeypatch.chdir(tmp_path)

    status, rc = _req(port, "POST", "/api/evals/run-configs", {"name": "rc1"})
    assert status == 201

    status, ec = _req(
        port,
        "POST",
        "/api/evals/eval-configs",
        {
            "name": "ec1",
            "rules": [
                {
                    "name": "rule1",
                    "checks": [
                        {"name": "c1", "evaluator": "exact_match", "expected": "pong"},
                        {"name": "c2", "evaluator": "exact_match", "expected": "nope"},
                    ],
                }
            ],
        },
    )
    assert status == 201

    status, evaluation = _req(
        port,
        "POST",
        "/api/evals/evaluations",
        {"name": "e1", "run_config_id": rc["id"], "eval_config_id": ec["id"]},
    )
    assert status == 201
    assert evaluation["last_run"] is None

    def fake_execute(self, max_steps=50):
        self.failure_error = None
        return [{"role": "assistant", "content": "pong"}]

    mocker.patch.object(AgentNode, "execute", fake_execute)

    status, run_body = _req(
        port, "POST", f"/api/evals/evaluations/{evaluation['id']}/run"
    )
    assert status == 202
    run_id = run_body["run_id"]

    # The run executes on a background thread (issue: a synchronous run
    # blocked the whole single-threaded HTTP server) — poll until the result
    # is written rather than assuming it's ready the instant the POST above
    # returns its 202.
    deadline = time.monotonic() + 5
    status, result = _req(
        port, "GET", f"/api/evals/evaluations/{evaluation['id']}/runs/{run_id}"
    )
    while status == 202 and time.monotonic() < deadline:
        time.sleep(0.05)
        status, result = _req(
            port, "GET", f"/api/evals/evaluations/{evaluation['id']}/runs/{run_id}"
        )
    assert status == 200
    assert len(result["scores"]) == 2
    check_ids = [c["id"] for c in ec["rules"][0]["checks"]]
    passed = {s["case_id"]: s["passed"] for s in result["scores"]}
    assert passed[check_ids[0]] is True
    assert passed[check_ids[1]] is False

    status, updated = _req(port, "GET", f"/api/evals/evaluations/{evaluation['id']}")
    assert status == 200
    assert updated["last_run"]["run_id"] == run_id

    # Listing still shows config intact (simulates "survives restart" via a fresh list).
    status, listing = _req(port, "GET", "/api/evals/evaluations")
    assert status == 200
    assert listing["evaluations"][0]["run_config_id"] == rc["id"]
    assert listing["evaluations"][0]["eval_config_id"] == ec["id"]


def test_delete_evaluation_leaves_configs_intact(server):
    port, _ = server
    status, rc = _req(port, "POST", "/api/evals/run-configs", {"name": "rc1"})
    status, ec = _req(
        port, "POST", "/api/evals/eval-configs", {"name": "ec1", "rules": []}
    )
    status, evaluation = _req(
        port,
        "POST",
        "/api/evals/evaluations",
        {"name": "e1", "run_config_id": rc["id"], "eval_config_id": ec["id"]},
    )
    assert status == 201

    status, _ = _req(port, "DELETE", f"/api/evals/evaluations/{evaluation['id']}")
    assert status == 204

    status, _ = _req(port, "GET", f"/api/evals/evaluations/{evaluation['id']}")
    assert status == 404

    status, _ = _req(port, "GET", f"/api/evals/run-configs/{rc['id']}")
    assert status == 200
    status, _ = _req(port, "GET", f"/api/evals/eval-configs/{ec['id']}")
    assert status == 200


def test_eval_config_routes_do_not_touch_existing_config_api(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/api/evals/config/datasets", {"id": "ds1", "case_ids": []}
    )
    assert status == 200
    assert body == {"id": "ds1", "version": 1, "case_ids": []}

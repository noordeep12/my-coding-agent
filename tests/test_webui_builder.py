"""Tests for webui/builder.py — Builder routes: node types, save/load, run control."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.pipeline.registry import CANONICAL_ORDER
from my_coding_agent.webui.builder import RunRegistry
from my_coding_agent.webui.server import _WebUIHandler
from my_coding_agent.webui.store import Store, default_db_path


@pytest.fixture()
def server(tmp_path):
    _WebUIHandler.base_dir = tmp_path
    _WebUIHandler.store = Store(default_db_path(tmp_path))
    _WebUIHandler.run_registry = RunRegistry()
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
    return resp.status, resp_body


def _canonical_graph():
    nodes = [
        {"id": f"n{i}", "type": t, "x": 0, "y": 0, "options": {}}
        for i, t in enumerate(CANONICAL_ORDER)
    ]
    edges = [
        {"from": nodes[i]["id"], "to": nodes[i + 1]["id"]}
        for i in range(len(nodes) - 1)
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "start": nodes[0]["id"],
        "end": nodes[-1]["id"],
    }


def test_builder_route_mounts_page(server):
    port, _ = server
    status, body = _req(port, "GET", "/builder")
    assert status == 200
    assert b"Pipeline Builder" in body


def test_node_types_lists_registered_types(server):
    port, _ = server
    status, body = _req(port, "GET", "/api/builder/node-types")
    assert status == 200
    names = [t["name"] for t in json.loads(body)]
    assert names == list(CANONICAL_ORDER)


def test_save_load_round_trip_preserves_graph(server):
    port, _ = server
    graph = _canonical_graph()
    status, body = _req(
        port, "POST", "/api/builder/pipelines", {"name": "demo", "graph": graph}
    )
    assert status == 201
    pipeline_id = json.loads(body)["id"]

    status, body = _req(port, "GET", f"/api/builder/pipelines/{pipeline_id}")
    assert status == 200
    loaded = json.loads(body)
    assert loaded["name"] == "demo"
    assert loaded["graph"] == graph


def test_list_and_delete_pipelines(server):
    port, _ = server
    graph = _canonical_graph()
    _req(port, "POST", "/api/builder/pipelines", {"name": "a", "graph": graph})
    status, body = _req(
        port, "POST", "/api/builder/pipelines", {"name": "b", "graph": graph}
    )
    pipeline_id = json.loads(body)["id"]

    status, body = _req(port, "GET", "/api/builder/pipelines")
    assert status == 200
    assert len(json.loads(body)) == 2

    status, _body = _req(port, "DELETE", f"/api/builder/pipelines/{pipeline_id}")
    assert status == 200
    status, body = _req(port, "GET", "/api/builder/pipelines")
    assert len(json.loads(body)) == 1


def test_update_pipeline(server):
    port, _ = server
    graph = _canonical_graph()
    status, body = _req(
        port, "POST", "/api/builder/pipelines", {"name": "a", "graph": graph}
    )
    pipeline_id = json.loads(body)["id"]

    status, _body = _req(
        port,
        "PUT",
        f"/api/builder/pipelines/{pipeline_id}",
        {"name": "renamed", "graph": graph},
    )
    assert status == 200
    status, body = _req(port, "GET", f"/api/builder/pipelines/{pipeline_id}")
    assert json.loads(body)["name"] == "renamed"


def test_run_rejects_non_runnable_graph(server):
    port, _ = server
    bad_graph = {
        "nodes": [{"id": "a", "type": "llm_call"}],
        "edges": [],
        "start": "a",
        "end": "a",
    }
    status, body = _req(
        port, "POST", "/api/builder/pipelines", {"name": "bad", "graph": bad_graph}
    )
    pipeline_id = json.loads(body)["id"]

    status, body = _req(
        port,
        "POST",
        f"/api/builder/pipelines/{pipeline_id}/run",
        {"task_prompt": "do something"},
    )
    assert status == 400
    assert "error" in json.loads(body)


def test_run_requires_task_prompt(server):
    port, _ = server
    graph = _canonical_graph()
    status, body = _req(
        port, "POST", "/api/builder/pipelines", {"name": "demo", "graph": graph}
    )
    pipeline_id = json.loads(body)["id"]

    status, body = _req(
        port, "POST", f"/api/builder/pipelines/{pipeline_id}/run", {"task_prompt": ""}
    )
    assert status == 400


def test_run_not_found_pipeline(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/api/builder/pipelines/nope/run", {"task_prompt": "x"}
    )
    assert status == 404


def test_run_status_for_unknown_run_id(server):
    port, _ = server
    status, body = _req(port, "GET", "/api/builder/runs/deadbeef00")
    assert status == 404


def test_stop_unknown_run_id(server):
    port, _ = server
    status, body = _req(port, "POST", "/api/builder/runs/deadbeef00/stop", {})
    assert status == 404

"""Tests for webui/server.py — shell page, mounted tabs, and state API."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.webui.server import NAV_TABS, _WebUIHandler
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


def _get(port, path):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _post(port, path, payload):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp_body = resp.read()
    conn.close()
    return resp.status, resp_body


def test_shell_route_returns_html_with_nav_tabs(server):
    port, _ = server
    status, body = _get(port, "/")
    assert status == 200
    html = body.decode()
    assert "<!DOCTYPE html>" in html
    for tab_id, _label in NAV_TABS:
        assert tab_id in html


def test_traces_route_mounts_trace_explorer(server):
    port, _ = server
    status, body = _get(port, "/traces")
    assert status == 200
    assert b"Trace Explorer" in body


def test_evals_route_mounts_eval_dashboard(server):
    port, _ = server
    status, body = _get(port, "/evals")
    assert status == 200
    assert b"Eval Dashboard" in body


def test_sessions_api_still_works_through_shell(server):
    port, _ = server
    status, body = _get(port, "/api/sessions")
    assert status == 200
    assert json.loads(body) == []


def test_evals_api_still_works_through_shell(server):
    port, _ = server
    status, body = _get(port, "/api/evals/runs")
    assert status == 200
    assert json.loads(body) == []


def test_state_round_trips_via_api(server):
    port, _ = server
    status, body = _get(port, "/api/webui/state")
    assert status == 200
    assert json.loads(body) == {}

    payload = {"route": "evals", "selection": {"evals": {"view": "runs"}}}
    status, body = _post(port, "/api/webui/state", payload)
    assert status == 200

    status, body = _get(port, "/api/webui/state")
    assert status == 200
    assert json.loads(body) == payload


def test_state_persists_across_server_restart(server, tmp_path):
    port, base = server
    payload = {"route": "traces", "selection": {"traces": {"session": "abc123"}}}
    _post(port, "/api/webui/state", payload)

    reopened = Store(default_db_path(base))
    assert reopened.get_ui_state("shell") == payload
    reopened.close()


def test_unknown_route_404(server):
    port, _ = server
    status, _body = _get(port, "/api/nope")
    assert status == 404

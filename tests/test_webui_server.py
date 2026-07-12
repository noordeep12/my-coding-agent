"""Tests for webui/server.py — shell page, mounted tabs, and state API."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.client import HTTPConnection
from http.server import HTTPServer, ThreadingHTTPServer

import pytest

from my_coding_agent.webui.server import NAV_TABS, _WebUIHandler, run_server
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


def _req_raw(port, method, path, raw_body):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(
        method, path, body=raw_body, headers={"Content-Type": "application/json"}
    )
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
    assert b"Evaluations" in body


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


def test_admin_route_returns_settings_form(server):
    port, _ = server
    status, body = _get(port, "/admin")
    assert status == 200
    assert b"LLM Connection Settings" in body


def test_admin_settings_api_round_trips_and_masks_key(server):
    port, _ = server
    status, body = _get(port, "/api/admin/settings")
    assert status == 200
    data = json.loads(body)
    assert "api_url" in data and "model" in data

    payload = {
        "api_url": "http://saved-host:9999/v1",
        "model": "saved-model",
        "api_key": "topsecret",  # pragma: allowlist secret
    }
    status, body = _post(port, "/api/admin/settings", payload)
    assert status == 200

    status, body = _get(port, "/api/admin/settings")
    data = json.loads(body)
    assert data["api_url"] == "http://saved-host:9999/v1"
    assert data["model"] == "saved-model"
    assert data["api_key"] == "********"
    assert b"topsecret" not in body


# ── Write-dispatch error paths ───────────────────────────────────────────────


def test_post_invalid_json_body_400(server):
    port, _ = server
    status, body = _req_raw(port, "POST", "/api/webui/state", b"not json")
    assert status == 400
    assert json.loads(body) == {"error": "invalid json"}


def test_post_state_non_dict_payload_400(server):
    port, _ = server
    status, body = _post(port, "/api/webui/state", ["not", "a", "dict"])
    assert status == 400
    assert json.loads(body) == {"error": "invalid payload"}


def test_post_admin_settings_non_dict_payload_400(server):
    port, _ = server
    status, body = _post(port, "/api/admin/settings", ["not", "a", "dict"])
    assert status == 400
    assert json.loads(body) == {"error": "invalid payload"}


def test_post_unknown_route_404(server):
    port, _ = server
    status, body = _post(port, "/api/nope", {"x": 1})
    assert status == 404
    assert json.loads(body) == {"error": "not found"}


def test_unhandled_write_method_on_eval_config_route_404(server):
    port, _ = server
    status, body = _req_raw(port, "PUT", "/api/evals/config/datasets", b"{}")
    assert status == 404
    assert json.loads(body) == {"error": "not found"}


# ── /api/session/{id} ────────────────────────────────────────────────────────


def _write_session_events(base_dir, session_id):
    session_dir = base_dir / session_id
    session_dir.mkdir(parents=True)
    events = [
        {
            "type": "session_start",
            "session_id": session_id,
            "label": "Test",
            "model": "gpt-4o-mini",
            "context_window": 8192,
            "started_at": "2026-01-01T10:00:00",
            "parent_session_id": None,
        },
        {
            "type": "llm_call",
            "call": 1,
            "kind": "main",
            "latency_s": 1.0,
            "prompt": 100,
            "completion": 50,
            "total": 150,
            "context_window": 8192,
            "messages": None,
            "response": {"content": "ok", "reasoning": "", "tool_calls": [], "raw": {}},
            "started_at": "2026-01-01T10:00:02",
        },
    ]
    with (session_dir / "events.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_session_api_returns_loaded_session(server):
    port, base = server
    _write_session_events(base, "abcd1234abcd")
    status, body = _get(port, "/api/session/abcd1234abcd")
    assert status == 200
    data = json.loads(body)
    assert data["session_id"] == "abcd1234abcd"


def test_session_api_rejects_malformed_session_id(server):
    port, _ = server
    status, body = _get(port, "/api/session/NOT-A-SID")
    assert status == 400
    assert json.loads(body) == {"error": "invalid session id"}


def test_session_api_loader_failure_500(server, monkeypatch):
    # The reader itself degrades gracefully on bad files, so simulate the
    # unexpected-failure case directly to prove it surfaces as a 500, not a
    # hung request or an unhandled traceback.
    def boom(events_path):
        raise RuntimeError("corrupt trace")

    monkeypatch.setattr("my_coding_agent.webui.server.load_session", boom)
    port, _ = server
    status, body = _get(port, "/api/session/aaaabbbbcccc")
    assert status == 500
    assert json.loads(body) == {"error": "corrupt trace"}


# ── run_server lifecycle ─────────────────────────────────────────────────────


def test_run_server_starts_and_stops_on_keyboard_interrupt(tmp_path, monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(ThreadingHTTPServer, "serve_forever", interrupt)
    run_server(host="127.0.0.1", port=0, base_dir=tmp_path)
    # The store bound to the handler class was closed on the way out.
    with pytest.raises(sqlite3.ProgrammingError):
        _WebUIHandler.store.get_ui_state("shell")

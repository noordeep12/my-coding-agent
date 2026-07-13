"""Tests for webui/evals/server.py routes — served offline, read-only."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from my_coding_agent.evals.results import build_run_result, write_run_result
from my_coding_agent.evals.schema import EvalScore
from my_coding_agent.webui.evals.server import (
    eval_dashboard_html,
    handle_eval_api_route,
)


class _EvalOnlyHandler(BaseHTTPRequestHandler):
    """Minimal handler exercising only the moved eval dashboard routes."""

    base_dir = None  # set as class attribute before serve_forever()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        eval_html = eval_dashboard_html(path)
        if eval_html is not None:
            self._send_html(eval_html)
        elif path.startswith("/api/evals/"):
            if not handle_eval_api_route(self, path, self.base_dir.resolve() / "evals"):
                self._send_json({"error": "not found"}, status=404)
        elif path == "/":
            self._send_html("<!DOCTYPE html><title>Trace Explorer</title>")
        else:
            self._send_json({"error": "not found"}, status=404)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


@pytest.fixture()
def server(tmp_path):
    """Spin up a real HTTP server on a random port, rooted at tmp_path."""
    _EvalOnlyHandler.base_dir = tmp_path
    httpd = None
    port = None
    for p in range(19900, 20000):
        try:
            httpd = HTTPServer(("127.0.0.1", p), _EvalOnlyHandler)
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


def _get(port, path):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _write_run(base_dir, *, pass_rate=1.0):
    result = build_run_result(
        dataset="ex@v1",
        scores=[
            EvalScore(case_id="c1", passed=True, metrics={"match": 1.0}, detail={})
        ],
        aggregate_metrics={"pass_rate": pass_rate},
    )
    write_run_result(result, root=base_dir / "evals")
    return result


class TestEvalDashboardHTML:
    def test_evals_route_returns_html(self, server):
        port, _ = server
        status, body = _get(port, "/evals")
        assert status == 200
        assert b"<!DOCTYPE html>" in body
        assert b"Evaluations" in body

    def test_evals_route_does_not_leak_into_trace_root(self, server):
        port, _ = server
        status, body = _get(port, "/")
        assert status == 200
        assert b"Evaluations" not in body


class TestEvalAPIRoutes:
    def test_runs_empty_when_no_results(self, server):
        port, _ = server
        status, body = _get(port, "/api/evals/runs")
        assert status == 200
        assert json.loads(body) == []

    def test_runs_lists_written_result(self, server):
        port, base_dir = server
        result = _write_run(base_dir)
        status, body = _get(port, "/api/evals/runs")
        data = json.loads(body)
        assert status == 200
        assert len(data) == 1
        assert data[0]["run_id"] == result.run_id
        assert data[0]["verdict"] == "pass"

    def test_run_detail_returns_breakdown(self, server):
        port, base_dir = server
        result = _write_run(base_dir)
        status, body = _get(port, f"/api/evals/runs/{result.run_id}")
        data = json.loads(body)
        assert status == 200
        assert data["summary"]["run_id"] == result.run_id
        assert data["cases"][0]["case_id"] == "c1"

    def test_run_detail_missing_run_is_404(self, server):
        port, _ = server
        status, body = _get(port, "/api/evals/runs/abc123")
        assert status == 404

    def test_run_detail_rejects_invalid_run_id(self, server):
        port, _ = server
        status, body = _get(port, "/api/evals/runs/../../etc")
        assert status in (400, 404)

    def test_datasets_empty_when_none_created(self, server):
        port, _ = server
        status, body = _get(port, "/api/evals/datasets")
        assert status == 200
        assert json.loads(body) == []

    def test_compare_is_a_labeled_stub(self, server):
        port, _ = server
        status, body = _get(port, "/api/evals/compare")
        data = json.loads(body)
        assert status == 200
        assert data["available"] is False

    def test_unknown_eval_route_is_404(self, server):
        port, _ = server
        status, body = _get(port, "/api/evals/nonsense")
        assert status == 404

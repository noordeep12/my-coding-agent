"""Tests for viewer/evals_server.py routes — served offline, read-only."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.evals.results import build_run_result, write_run_result
from my_coding_agent.evals.schema import EvalScore
from my_coding_agent.viewer.server import _TraceHandler


@pytest.fixture()
def server(tmp_path):
    """Spin up a real HTTP server on a random port, rooted at tmp_path."""
    _TraceHandler.base_dir = tmp_path
    httpd = None
    port = None
    for p in range(19900, 20000):
        try:
            httpd = HTTPServer(("127.0.0.1", p), _TraceHandler)
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
        assert b"Eval Dashboard" in body

    def test_evals_route_does_not_leak_into_trace_root(self, server):
        port, _ = server
        status, body = _get(port, "/")
        assert status == 200
        assert b"Eval Dashboard" not in body


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

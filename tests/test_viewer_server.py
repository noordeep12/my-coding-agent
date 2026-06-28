"""Tests for viewer/server.py — HTTP routes and path-traversal guard."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

from my_coding_agent.viewer.server import _SID_RE, EMBEDDED_HTML, _TraceHandler

# ── SID regex ────────────────────────────────────────────────────────────────


class TestSidRegex:
    def test_valid_8_hex(self):
        assert _SID_RE.match("abcdef01")

    def test_valid_64_hex(self):
        assert _SID_RE.match("a" * 64)

    def test_rejects_7_chars(self):
        assert not _SID_RE.match("abcdef0")

    def test_rejects_uppercase(self):
        assert not _SID_RE.match("ABCDEF01")

    def test_rejects_dots(self):
        assert not _SID_RE.match("../../etc")

    def test_rejects_slash(self):
        assert not _SID_RE.match("aabb/ccdd")

    def test_rejects_65_chars(self):
        assert not _SID_RE.match("a" * 65)


# ── Live server fixture ────────────────────────────────────────────────────────


@pytest.fixture()
def server(tmp_path):
    """Spin up a real HTTP server on a random port in a background thread."""
    _TraceHandler.base_dir = tmp_path
    httpd = None
    port = None
    for p in range(19800, 19900):
        try:
            from http.server import HTTPServer

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


# ── Route tests ────────────────────────────────────────────────────────────────


class TestRoutes:
    def test_root_returns_html(self, server):
        port, _ = server
        status, body = _get(port, "/")
        assert status == 200
        assert b"<!DOCTYPE html>" in body

    def test_root_contains_embedded_html(self, server):
        port, _ = server
        status, body = _get(port, "/")
        assert EMBEDDED_HTML[:50].encode() in body

    def test_sessions_empty_dir(self, server):
        port, _ = server
        status, body = _get(port, "/api/sessions")
        assert status == 200
        assert json.loads(body) == []

    def test_unknown_route_404(self, server):
        port, _ = server
        status, body = _get(port, "/api/nope")
        assert status == 404

    def test_session_invalid_id_400(self, server):
        port, _ = server
        # Non-hex characters in session ID → 400 (regex rejects before path join)
        status, body = _get(port, "/api/session/not-a-hex-id!")
        assert status == 400

    def test_session_valid_id_missing_dir(self, server):
        port, tmp_path = server
        # valid hex ID but no directory → load_session falls back, returns 200
        status, body = _get(port, "/api/session/aabbccdd1234abcd")
        # either 200 (fallback session) or 500 (exception); must not be 400/404
        assert status in (200, 500)

    def test_session_uppercase_id_rejected(self, server):
        port, _ = server
        status, body = _get(port, "/api/session/AABBCCDD1234ABCD")
        assert status == 400

    def test_sessions_with_data(self, server, tmp_path):
        port, base = server
        sid = "aabbccdd1234abcd"
        sdir = base / sid
        sdir.mkdir()
        events = [
            json.dumps(
                {
                    "type": "session_start",
                    "session_id": sid,
                    "label": "T",
                    "model": "m",
                    "context_window": 8192,
                    "started_at": "2026-01-01T00:00:00",
                    "parent_session_id": None,
                }
            ),
            json.dumps(
                {
                    "type": "session_end",
                    "stop_reason": "stop",
                    "steps": 1,
                    "elapsed_s": 1.0,
                    "ended_at": "2026-01-01T00:00:01",
                }
            ),
        ]
        (sdir / "events.jsonl").write_text("\n".join(events), encoding="utf-8")
        status, body = _get(port, "/api/sessions")
        assert status == 200
        rows = json.loads(body)
        assert any(r["session_id"] == sid for r in rows)

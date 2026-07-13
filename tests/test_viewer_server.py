"""Tests for viewer/server.py — HTTP routes and path-traversal guard."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

from my_coding_agent.viewer.server import (
    _SID_RE,
    EMBEDDED_HTML,
    _TraceHandler,
    run_server,
)

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


@pytest.fixture
def server(tmp_path):
    """Spin up a real HTTP server on a random port in a background thread."""
    _TraceHandler.base_dir = tmp_path
    httpd = None
    port = None
    for p in range(19800, 19900):
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

    def test_tree_group_renders_its_own_ctx_state_summary(self, server):
        """Regression: a node with nested children (e.g. a `delegate` tool_call,
        which always nests its subagent session root under it) must still show
        its own ctx-window contribution ("+N tool") — TreeGroup previously
        dropped it entirely, unlike TreeLeaf which renders leaf (childless)
        tool_call nodes such as bash.
        """
        port, _ = server
        status, body = _get(port, "/")
        html = body.decode()
        tree_group_src = html[
            html.index("function TreeGroup") : html.index("function TreeLeaf")
        ]
        assert "addedParts(node.ctx_state)" in tree_group_src
        assert "tleaf-sub" in tree_group_src

    def test_retired_tokens_open_a_before_after_modal(self, server):
        """The right-pane "retired" figure is clickable only when the node
        carries retirements, opening RetirementModal to show each retired
        message's original content next to the stub that replaced it — the
        "added" figure stays plain text, never a button, per product intent
        (added content is already visible in the node's own output).
        """
        port, _ = server
        status, body = _get(port, "/")
        html = body.decode()
        assert "function RetirementModal" in html
        assert "retire-before" in html and "retire-after" in html
        ctx_card_src = html[html.index("function CtxCard") :]
        assert "retirements.length?()=>setShowRetired(true):null" in ctx_card_src
        assert "ctx-delta-add" in ctx_card_src
        # The added span must never gain an onClick — it stays plain text.
        added_span = ctx_card_src[
            ctx_card_src.index('class="ctx-delta-add"') : ctx_card_src.index(
                "ctx-delta-rem"
            )
        ]
        assert "onClick" not in added_span

    def test_multiline_bash_badge_detection_source(self, server):
        """Regression: bash-stdin-delivery — the multi-line badge must be
        detected from recorded args alone (non-empty `stdin`, or a newline in
        `command`), so old traces without `stdin` still badge via the
        newline-in-command signal, and plain single-line calls stay unbadged.
        """
        port, _ = server
        status, body = _get(port, "/")
        html = body.decode()
        detect_src = html[
            html.index("function isMultilineBashCall") : html.index(
                "function nodeBadges"
            )
        ]
        assert "args.stdin" in detect_src
        assert "args.command" in detect_src
        assert "includes('\\n')" in detect_src

        node_badges_src = html[
            html.index("function nodeBadges") : html.index("const treeBadges")
        ]
        assert "isMultilineBashCall" in node_badges_src

    def test_report_provenance_badge_source(self, server):
        """The report node's badge distinguishes free/paid/unknown provenance
        at a glance (D3: unknown, never a guessed path, when source is absent).
        """
        port, _ = server
        status, body = _get(port, "/")
        html = body.decode()
        node_badges_src = html[
            html.index("function nodeBadges") : html.index("const treeBadges")
        ]
        assert "node.type==='report'" in node_badges_src
        assert "a.source==='verbatim'" in node_badges_src
        assert "a.source==='summarizer'" in node_badges_src

    def test_refusal_badge_rendered_tree_and_detail(self, server):
        """A refused tool_call node shows a distinct `refusal-tag` in both the
        tree row (TreeLeaf/TreeGroup) and the detail header, alongside — never
        replacing — loop/anomaly, and the stats strip renders a refusal
        count when the session's analytics carry one (issue #124).
        """
        port, _ = server
        status, body = _get(port, "/")
        html = body.decode()
        assert "refusal-tag" in html
        assert "node.refusal_flag" in html
        assert "a.refusal_count" in html

    def test_refusal_detail_block_renders_reason_and_reference_links(self, server):
        port, _ = server
        status, body = _get(port, "/")
        html = body.decode()
        detail_src = html[
            html.index("function RefusalDetail") : html.index("function ToolResult")
        ]
        assert "refusal.reason" in detail_src
        assert "refusal.safer_alternative" in detail_src
        assert "r.url" in detail_src
        assert "r.standard_id" in detail_src

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
        status, body = _get(port, "/api/sessions/not-a-hex-id!")
        assert status == 400

    def test_session_valid_id_missing_dir(self, server):
        port, tmp_path = server
        # valid hex ID but no directory → load_session falls back, returns 200
        status, body = _get(port, "/api/sessions/aabbccdd1234abcd")
        # either 200 (fallback session) or 500 (exception); must not be 400/404
        assert status in (200, 500)

    def test_session_uppercase_id_rejected(self, server):
        port, _ = server
        status, body = _get(port, "/api/sessions/AABBCCDD1234ABCD")
        assert status == 400

    def test_singular_session_path_not_served(self, server):
        port, _ = server
        status, body = _get(port, "/api/session/aabbccdd1234abcd")
        assert status == 404

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

    def _write_session(self, base, sid, events):
        sdir = base / sid
        sdir.mkdir()
        (sdir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
        return sdir

    def test_session_payload_carries_verdict_for_scored_session(self, server):
        port, base = server
        sid = "aabbccdd1234abce"
        events = [
            {
                "type": "session_start",
                "session_id": sid,
                "label": "T",
                "model": "m",
                "context_window": 8192,
                "started_at": "2026-01-01T00:00:00",
                "parent_session_id": None,
            },
            {
                "type": "session_end",
                "stop_reason": "stop",
                "steps": 1,
                "elapsed_s": 1.0,
                "ended_at": "2026-01-01T00:00:01",
            },
        ]
        sdir = self._write_session(base, sid, events)
        verdict = {
            "run_id": "run-1",
            "case_id": "case-1",
            "passed": True,
            "metrics": {"score": 1.0},
            "detail": "great",
            "result_path": "/tmp/result.json",
        }
        (sdir / "verdict.json").write_text(json.dumps(verdict))
        status, body = _get(port, f"/api/sessions/{sid}")
        assert status == 200
        payload = json.loads(body)
        assert payload["verdict"] == verdict
        assert payload["session_id"] == sid

    def test_session_payload_has_no_verdict_for_unscored_session(self, server):
        port, base = server
        sid = "aabbccdd1234abcf"
        events = [
            {
                "type": "session_start",
                "session_id": sid,
                "label": "T",
                "model": "m",
                "context_window": 8192,
                "started_at": "2026-01-01T00:00:00",
                "parent_session_id": None,
            },
            {
                "type": "session_end",
                "stop_reason": "stop",
                "steps": 1,
                "elapsed_s": 1.0,
                "ended_at": "2026-01-01T00:00:01",
            },
        ]
        self._write_session(base, sid, events)
        status, body = _get(port, f"/api/sessions/{sid}")
        assert status == 200
        payload = json.loads(body)
        assert payload["verdict"] is None
        assert payload["session_id"] == sid


# ── run_server lifecycle ─────────────────────────────────────────────────────


def test_run_server_starts_and_stops_on_keyboard_interrupt(tmp_path, monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(HTTPServer, "serve_forever", interrupt)
    run_server(host="127.0.0.1", port=0, base_dir=tmp_path)
    assert _TraceHandler.base_dir == tmp_path

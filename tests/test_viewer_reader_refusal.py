"""Tests for the refusal_flag / refusal_count viewer surfacing (issue #124).

A refused tool_call event's own recorded envelope carries ``metadata.refusal``
(the exact fact the model itself saw); the reader flags that node directly —
no positional back-walk, unlike anomaly's streak matching — and counts it in
analytics. Sessions with no refusals, and pre-change traces with no
``metadata.refusal`` key at all, must render byte-identically.
"""

import json
from pathlib import Path

from my_coding_agent.viewer.reader import load_session


def _ev(type: str, **kw):
    return {"type": type, **kw}


def _write_events(path: Path, events: list) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def _refused_result() -> str:
    env = {
        "schema_version": 1,
        "tool": "bash",
        "ok": False,
        "output": "",
        "error": (
            "Refused (not a failure): 'rm -rf /' — destroys the filesystem. "
            "Reference: NIST SP 800-53 SI-7 "
            "(https://csrc.nist.gov/controls/sp800-53/rev5/si-7). "
            "Safer alternative: scope the delete to a subdirectory."
        ),
        "metadata": {
            "reason": "refused",
            "refusal": {
                "rule_id": "rm_root_class_path",
                "reason": "destroys the filesystem",
                "references": [
                    {
                        "standard_id": "NIST SP 800-53 SI-7",
                        "url": "https://csrc.nist.gov/controls/sp800-53/rev5/si-7",
                    }
                ],
                "safer_alternative": "scope the delete to a subdirectory",
            },
        },
    }
    return json.dumps(env)


def _session_events(session_id: str, *, with_refusal: bool) -> list:
    tool_result = _refused_result() if with_refusal else json.dumps(
        {
            "schema_version": 1,
            "tool": "bash",
            "ok": True,
            "output": "hi",
            "error": None,
            "metadata": {},
        }
    )
    return [
        _ev(
            "session_start",
            session_id=session_id,
            label="Test",
            model="gpt-4o-mini",
            context_window=8192,
            started_at="2026-01-01T10:00:00",
            parent_session_id=None,
        ),
        _ev(
            "llm_call",
            call=1,
            kind="main",
            latency_s=1.0,
            prompt=100,
            completion=50,
            total=150,
            context_window=8192,
            messages=None,
            response={"content": "", "reasoning": "", "tool_calls": [], "raw": {}},
            started_at="2026-01-01T10:00:01",
        ),
        _ev(
            "tool_call",
            name="bash",
            args={"command": "rm -rf /"},
            result=tool_result,
            ok=not with_refusal,
            latency_s=0.01,
            started_at="2026-01-01T10:00:02",
        ),
        _ev(
            "session_end",
            stop_reason="stop",
            steps=1,
            elapsed_s=0.5,
            ended_at="2026-01-01T10:00:03",
        ),
    ]


def _load(tmp_path, sid, events):
    sdir = tmp_path / sid
    sdir.mkdir()
    ep = sdir / "events.jsonl"
    _write_events(ep, events)
    return load_session(ep)


class TestRefusalFlag:
    def test_refused_tool_call_is_flagged(self, tmp_path):
        sid = "aabbccdd1111"
        session = _load(tmp_path, sid, _session_events(sid, with_refusal=True))
        tool_nodes = [n for n in session.nodes.values() if n.type == "tool_call"]
        assert len(tool_nodes) == 1
        assert tool_nodes[0].refusal_flag is True

    def test_refusal_count_in_analytics(self, tmp_path):
        sid = "aabbccdd1111"
        session = _load(tmp_path, sid, _session_events(sid, with_refusal=True))
        assert session.analytics["refusal_count"] == 1

    def test_session_without_refusals_is_unflagged(self, tmp_path):
        sid = "aabbccdd2222"
        session = _load(tmp_path, sid, _session_events(sid, with_refusal=False))
        tool_nodes = [n for n in session.nodes.values() if n.type == "tool_call"]
        assert not any(n.refusal_flag for n in tool_nodes)
        assert session.analytics["refusal_count"] == 0

    def test_pre_change_trace_with_no_metadata_key_renders_unflagged(self, tmp_path):
        """A tool_call event recorded before this change has no
        ``metadata.refusal`` key at all (not even absent-but-present) — the
        reader must not error and must simply leave the node unflagged."""
        sid = "aabbccdd3333"
        events = _session_events(sid, with_refusal=False)
        legacy_result = json.dumps({"ok": True, "stdout": "hi", "stderr": ""})
        for ev in events:
            if ev.get("type") == "tool_call":
                ev["result"] = legacy_result
        session = _load(tmp_path, sid, events)
        tool_nodes = [n for n in session.nodes.values() if n.type == "tool_call"]
        assert not any(n.refusal_flag for n in tool_nodes)
        assert session.analytics["refusal_count"] == 0

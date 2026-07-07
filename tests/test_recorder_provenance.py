"""Regression tests for Recorder.record_provenance (issue #128).

Each row must be answerable with a single ``json.loads`` (one JSON object per
line), a run with no ingestion must emit no provenance rows, and a
pre-change ``events.jsonl`` (no provenance rows at all) must still load
unchanged — additive-only, matching the migration plan in design.md.
"""

from __future__ import annotations

import json

from my_coding_agent.observability.recorder import PROVENANCE, Recorder


def _make_recorder(tmp_path):
    sid = "aabbccdd"
    sdir = tmp_path / sid
    sdir.mkdir()
    return Recorder(session_id=sid, session_dir=sdir), sdir / "events.jsonl"


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRecordProvenance:
    def test_emits_one_json_object_per_line(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_provenance(
            kind="mark", tool_name="fetch_web", reason="tagged untrusted", step=1
        )
        events = _read_events(path)
        assert len(events) == 1
        assert events[0]["type"] == PROVENANCE
        assert events[0]["kind"] == "mark"
        assert events[0]["tool_name"] == "fetch_web"
        assert events[0]["reason"] == "tagged untrusted"
        assert events[0]["step"] == 1
        assert "started_at" in events[0]

    def test_reduction_refusal_row_shape(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        rec.record_provenance(
            kind="reduction_refusal",
            tool_name="bash",
            reason="clone-and-build reduction",
            step=4,
        )
        events = _read_events(path)
        assert events[0]["kind"] == "reduction_refusal"

    def test_no_ingestion_run_emits_no_provenance_rows(self, tmp_path):
        """A run that never calls record_provenance leaves events.jsonl either
        absent or free of any provenance-type row — the no-ingestion
        byte-identical guarantee at the recorder layer."""
        rec, path = _make_recorder(tmp_path)
        rec.record_token_tracking(
            step=1,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            ctx_pct=1.0,
            context_window=8000,
        )
        events = _read_events(path)
        assert all(e["type"] != PROVENANCE for e in events)

    def test_pre_change_trace_without_provenance_rows_still_parses(self, tmp_path):
        """A trace written before this change (no provenance rows at all)
        parses exactly as it did before — provenance is purely additive."""
        sid = "aabbccdd"
        sdir = tmp_path / sid
        sdir.mkdir()
        path = sdir / "events.jsonl"
        pre_change_row = {"type": "session_start", "session_id": sid}
        path.write_text(json.dumps(pre_change_row) + "\n")
        events = _read_events(path)
        assert events == [pre_change_row]

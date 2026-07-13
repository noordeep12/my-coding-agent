"""Tests for the deterministic, LLM-free sum-check (D4)."""

from __future__ import annotations

import json

from my_coding_agent.viewer.sumcheck import check_session, check_tree

_TOKENS = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _write_session(
    session_dir,
    session_id,
    llm_calls,
    by_kind,
    grand_total,
    descendants=None,
    report_event=None,
    write_session_data=True,
):
    session_dir.mkdir(parents=True, exist_ok=True)
    if write_session_data:
        data = {
            "session_id": session_id,
            "llm_calls": llm_calls,
            "rollup": {
                "session_id": session_id,
                "by_kind": by_kind,
                "descendants": descendants or [],
                "grand_total": grand_total,
            },
        }
        (session_dir / "session_data.json").write_text(json.dumps(data))
    if report_event is not None:
        (session_dir / "events.jsonl").write_text(json.dumps(report_event) + "\n")


def _call(kind, prompt=10, completion=5, total=15):
    return {"kind": kind, "prompt": prompt, "completion": completion, "total": total}


class TestCheckSessionArithmetic:
    def test_consistent_session_passes(self, tmp_path):
        sdir = tmp_path / "s1"
        _write_session(
            sdir,
            "s1",
            [_call("main")],
            {"main": dict(_TOKENS)},
            dict(_TOKENS),
            report_event={"type": "report", "source": "verbatim"},
        )
        result = check_session(sdir)
        assert result.status == "pass"
        assert result.reasons == []

    def test_by_kind_mismatch_fails_naming_kind(self, tmp_path):
        sdir = tmp_path / "s2"
        _write_session(
            sdir,
            "s2",
            [_call("main")],
            {
                "main": {
                    "prompt_tokens": 999,
                    "completion_tokens": 999,
                    "total_tokens": 999,
                }
            },
            {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 999},
            report_event={"type": "report", "source": "verbatim"},
        )
        result = check_session(sdir)
        assert result.status == "fail"
        assert "main" in result.reasons[0]

    def test_grand_total_mismatch_fails(self, tmp_path):
        sdir = tmp_path / "s3"
        _write_session(
            sdir,
            "s3",
            [_call("main")],
            {"main": dict(_TOKENS)},
            {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 999},
            report_event={"type": "report", "source": "verbatim"},
        )
        result = check_session(sdir)
        assert result.status == "fail"
        assert any("grand_total" in r for r in result.reasons)

    def test_missing_session_data_is_unverifiable(self, tmp_path):
        sdir = tmp_path / "crashed"
        sdir.mkdir()
        result = check_session(sdir)
        assert result.status == "unverifiable"

    def test_descendant_grand_total_included(self, tmp_path):
        sdir = tmp_path / "parent"
        child_total = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        expected_grand = {k: _TOKENS[k] + child_total[k] for k in _TOKENS}
        _write_session(
            sdir,
            "parent",
            [_call("main")],
            {"main": dict(_TOKENS)},
            expected_grand,
            descendants=[{"session_id": "child", "grand_total": child_total}],
            report_event={"type": "report", "source": "verbatim"},
        )
        result = check_session(sdir)
        assert result.status == "pass"


class TestReportProvenanceInvariant:
    def test_verbatim_with_no_report_kind_row_passes(self, tmp_path):
        sdir = tmp_path / "s"
        _write_session(
            sdir,
            "s",
            [_call("main")],
            {"main": dict(_TOKENS)},
            dict(_TOKENS),
            report_event={"type": "report", "source": "verbatim"},
        )
        assert check_session(sdir).status == "pass"

    def test_verbatim_with_report_kind_row_fails(self, tmp_path):
        sdir = tmp_path / "s"
        calls = [_call("main"), _call("report")]
        by_kind = {"main": dict(_TOKENS), "report": dict(_TOKENS)}
        grand = {k: _TOKENS[k] * 2 for k in _TOKENS}
        _write_session(
            sdir,
            "s",
            calls,
            by_kind,
            grand,
            report_event={"type": "report", "source": "verbatim"},
        )
        result = check_session(sdir)
        assert result.status == "fail"
        assert any("verbatim" in r for r in result.reasons)

    def test_summarizer_with_one_report_kind_row_passes(self, tmp_path):
        sdir = tmp_path / "s"
        calls = [_call("main"), _call("report")]
        by_kind = {"main": dict(_TOKENS), "report": dict(_TOKENS)}
        grand = {k: _TOKENS[k] * 2 for k in _TOKENS}
        _write_session(
            sdir,
            "s",
            calls,
            by_kind,
            grand,
            report_event={"type": "report", "source": "summarizer"},
        )
        assert check_session(sdir).status == "pass"

    def test_fallback_with_zero_report_kind_rows_fails(self, tmp_path):
        sdir = tmp_path / "s"
        _write_session(
            sdir,
            "s",
            [_call("main")],
            {"main": dict(_TOKENS)},
            dict(_TOKENS),
            report_event={"type": "report", "source": "fallback"},
        )
        result = check_session(sdir)
        assert result.status == "fail"
        assert any("fallback" in r for r in result.reasons)

    def test_pre_provenance_report_is_unverifiable_not_failed(self, tmp_path):
        sdir = tmp_path / "s"
        _write_session(
            sdir,
            "s",
            [_call("main")],
            {"main": dict(_TOKENS)},
            dict(_TOKENS),
            report_event={"type": "report"},
        )
        result = check_session(sdir)
        assert result.status == "unverifiable"
        assert any("provenance" in r for r in result.reasons)

    def test_no_report_event_is_unaffected(self, tmp_path):
        """A session that never delegated (no report event at all) is a
        pass/fail on arithmetic alone; the provenance invariant is silent.
        """
        sdir = tmp_path / "s"
        _write_session(
            sdir, "s", [_call("main")], {"main": dict(_TOKENS)}, dict(_TOKENS)
        )
        result = check_session(sdir)
        assert result.status == "pass"


class TestCheckTree:
    def test_recurses_into_descendants(self, tmp_path):
        child_total = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        parent_grand = {k: _TOKENS[k] + child_total[k] for k in _TOKENS}
        _write_session(
            tmp_path / "parent",
            "parent",
            [_call("main")],
            {"main": dict(_TOKENS)},
            parent_grand,
            descendants=[{"session_id": "child", "grand_total": child_total}],
            report_event={"type": "report", "source": "verbatim"},
        )
        _write_session(
            tmp_path / "child",
            "child",
            [_call("main", prompt=100, completion=20, total=120)],
            {"main": child_total},
            child_total,
            report_event={"type": "report", "source": "verbatim"},
        )
        results = check_tree(tmp_path, "parent")
        assert {r.session_id for r in results} == {"parent", "child"}
        assert all(r.status == "pass" for r in results)

    def test_doctored_child_surfaces_as_failure_in_the_tree(self, tmp_path):
        child_total = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        parent_grand = {k: _TOKENS[k] + child_total[k] for k in _TOKENS}
        _write_session(
            tmp_path / "parent",
            "parent",
            [_call("main")],
            {"main": dict(_TOKENS)},
            parent_grand,
            descendants=[{"session_id": "child", "grand_total": child_total}],
            report_event={"type": "report", "source": "verbatim"},
        )
        _write_session(
            tmp_path / "child",
            "child",
            [_call("main", prompt=1, completion=1, total=2)],  # doctored: too low
            {"main": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            child_total,
            report_event={"type": "report", "source": "verbatim"},
        )
        results = check_tree(tmp_path, "parent")
        by_id = {r.session_id: r for r in results}
        assert by_id["parent"].status == "pass"
        assert by_id["child"].status == "fail"

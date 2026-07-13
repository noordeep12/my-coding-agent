"""Tests for the task-level cost section of print_run_summary (task 3)."""

from __future__ import annotations

import re

from my_coding_agent.utils.terminal_ui import print_run_summary

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(capsys, **kwargs):
    defaults = dict(
        steps=1,
        max_steps=5,
        stop_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    defaults.update(kwargs)
    print_run_summary(**defaults)
    out = capsys.readouterr().err
    return _ANSI_RE.sub("", out)


def test_no_rollup_section_without_descendants(capsys):
    plain = _capture(capsys, rollup=None)
    assert "SUBAGENTS" not in plain
    assert "TASK TOTAL" not in plain


def test_no_rollup_section_when_rollup_has_no_descendants(capsys):
    rollup = {
        "by_kind": {
            "main": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        },
        "descendants": [],
        "grand_total": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    plain = _capture(capsys, rollup=rollup)
    assert "SUBAGENTS" not in plain


def test_rollup_section_shows_own_vs_rolled_up_and_per_subagent_line(capsys):
    rollup = {
        "by_kind": {
            "main": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        },
        "descendants": [
            {
                "session_id": "child-abc",
                "elapsed_s": 12.3,
                "steps": 4,
                "grand_total": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            }
        ],
        "grand_total": {
            "prompt_tokens": 110,
            "completion_tokens": 25,
            "total_tokens": 135,
        },
    }
    plain = _capture(capsys, rollup=rollup)
    assert "SUBAGENTS" in plain
    assert "TASK TOTAL (own)" in plain
    assert "TASK TOTAL (rolled up)" in plain
    assert "15 tok" in plain  # own total
    assert "135 tok" in plain  # rolled-up total
    assert "child-abc" in plain
    assert "120" in plain  # subagent's own grand total
    assert "12.3s" in plain


def test_rollup_section_distinguishes_free_and_paid_reports(capsys):
    """A clean-finish (verbatim) child is marked free; a cutoff (summarizer)
    child is marked paid — using the report source handed up in D5.
    """
    rollup = {
        "by_kind": {},
        "descendants": [
            {
                "session_id": "child-clean",
                "elapsed_s": 1.0,
                "steps": 2,
                "grand_total": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "report_source": "verbatim",
            },
            {
                "session_id": "child-cutoff",
                "elapsed_s": 2.0,
                "steps": 3,
                "grand_total": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
                "report_source": "summarizer",
            },
        ],
        "grand_total": {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
        },
    }
    plain = _capture(capsys, rollup=rollup)
    lines = plain.splitlines()
    clean_line = next(line for line in lines if "child-clean" in line)
    cutoff_line = next(line for line in lines if "child-cutoff" in line)
    assert "free report" in clean_line
    assert "paid report" in cutoff_line

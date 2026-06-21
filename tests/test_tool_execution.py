"""Tests for tool-execution helpers (summary extraction + output schema)."""

import pytest

from my_coding_agent.tool_execution import (
    TOOL_SCHEMA_VERSION,
    _extract_summary,
    build_tool_result,
    validate_tool_result,
)


def test_extract_summary_prefers_summary_tags():
    content = (
        "Here's a thinking process:\n1. analyze...\n2. write it\n"
        "<summary>Command succeeded (exit 0); fetched the KEV catalog.</summary>"
    )
    assert _extract_summary(content) == (
        "Command succeeded (exit 0); fetched the KEV catalog."
    )


def test_extract_summary_strips_think_blocks_when_no_summary_tag():
    content = "<think>let me reason about this</think>The build passed cleanly."
    assert _extract_summary(content) == "The build passed cleanly."


def test_extract_summary_passes_through_clean_content():
    assert _extract_summary("  Just the summary.  ") == "Just the summary."


# --- canonical tool-output schema ---------------------------------------------


def test_build_tool_result_has_full_envelope():
    env = build_tool_result("bash", True, "hi", metadata={"exit_code": 0})
    assert env == {
        "schema_version": TOOL_SCHEMA_VERSION,
        "tool": "bash",
        "ok": True,
        "output": "hi",
        "error": None,
        "metadata": {"exit_code": 0},
    }
    validate_tool_result(env)  # a built envelope always conforms


def test_validate_tool_result_accepts_minimal_failure():
    env = build_tool_result("read_file", False, error="file not found")
    assert validate_tool_result(env) is env
    assert env["metadata"] == {}


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        {"tool": "x"},  # missing keys
        {**build_tool_result("x", True), "ok": "yes"},  # ok not bool
        {**build_tool_result("x", True), "output": 123},  # output not str
        {**build_tool_result("x", True), "metadata": []},  # metadata not dict
    ],
)
def test_validate_tool_result_rejects_malformed(bad):
    with pytest.raises(ValueError):
        validate_tool_result(bad)

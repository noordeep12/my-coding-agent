"""Tests for the testable logic in logger.py.

The module is mostly terminal-UI rendering (print-only) which is low-value to
assert. These tests target the units with assertable output: the tool-count
summary label, the custom log-level registration, the TeeStream ANSI-stripping
fan-out, the git-branch fallback, and the custom-level logger methods.
"""

import io
import logging

from my_coding_agent import logger as lg

# --- _tool_count_label -------------------------------------------------------


def test_tool_count_label_empty():
    assert lg._tool_count_label([]) == "0"


def test_tool_count_label_all_ok():
    records = [{"ok": True}, {"ok": True}]
    assert lg._tool_count_label(records) == "2 (2 ok)"


def test_tool_count_label_mixed_ok_and_failed():
    records = [{"ok": True}, {"ok": False}]
    assert lg._tool_count_label(records) == "2 (1 ok, 1 failed)"


def test_tool_count_label_counts_skipped_separately():
    records = [
        {"ok": True},
        {"ok": False, "status": "skipped"},
        {"ok": False},
    ]
    # 1 ok, 1 failed (the plain False), 1 skipped.
    assert lg._tool_count_label(records) == "3 (1 ok, 1 failed, 1 skipped)"


# --- custom level numbers / _register_level_names ----------------------------


def test_custom_level_numbers_are_between_debug_and_info():
    assert logging.DEBUG < lg.TOOL < logging.INFO < lg.API < lg.LLM


def test_register_level_names_maps_numbers_to_names():
    lg._register_level_names()
    assert logging.getLevelName(lg.TOOL) == "TOOL"
    assert logging.getLevelName(lg.API) == "API"
    assert logging.getLevelName(lg.LLM) == "LLM"


# --- _TeeStream --------------------------------------------------------------


def test_teestream_write_fans_out_and_strips_ansi():
    orig, plain, colored = io.StringIO(), io.StringIO(), io.StringIO()
    tee = lg._TeeStream(orig, plain, colored)
    colored_text = "\x1b[31mred\x1b[0m text"

    n = tee.write(colored_text)

    assert n == len(colored_text)            # returns count written
    assert orig.getvalue() == colored_text    # original keeps the ANSI codes
    assert colored.getvalue() == colored_text  # colored file keeps them too
    assert plain.getvalue() == "red text"     # plain file is stripped of ANSI


def test_teestream_flush_propagates():
    class _Track(io.StringIO):
        flushed = False

        def flush(self):
            self.flushed = True

    orig, plain, colored = _Track(), _Track(), _Track()
    lg._TeeStream(orig, plain, colored).flush()
    assert orig.flushed and plain.flushed and colored.flushed


def test_teestream_getattr_delegates_to_original():
    orig = io.StringIO()
    orig.custom_attr = "from original"  # type: ignore[attr-defined]
    tee = lg._TeeStream(orig, io.StringIO(), io.StringIO())
    assert tee.custom_attr == "from original"


# --- _git_branch -------------------------------------------------------------


def test_git_branch_returns_output(mocker):
    mocker.patch(
        "my_coding_agent.logger.subprocess.check_output",
        return_value=b"feature-x\n",
    )
    assert lg._git_branch() == "feature-x"


def test_git_branch_falls_back_to_unknown_on_error(mocker):
    mocker.patch(
        "my_coding_agent.logger.subprocess.check_output",
        side_effect=OSError("git missing"),
    )
    assert lg._git_branch() == "unknown"


# --- get_logger custom methods -----------------------------------------------


def test_get_logger_has_custom_level_methods():
    log = lg.get_logger("test-logger-custom-methods")
    assert callable(log.tool)
    assert callable(log.api)
    assert callable(log.llm)


def test_get_logger_emits_at_custom_level(caplog):
    log = lg.get_logger("test-logger-emit", level=logging.DEBUG)
    with caplog.at_level(lg.TOOL, logger="test-logger-emit"):
        log.tool("a tool message")
    assert "a tool message" in caplog.text

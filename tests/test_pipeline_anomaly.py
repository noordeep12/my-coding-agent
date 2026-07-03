"""Tests for pipeline/anomaly.py: error_signature and trailing_streak."""

from __future__ import annotations

import pytest

from my_coding_agent.pipeline.anomaly import (
    STREAK_THRESHOLD,
    error_signature,
    trailing_streak,
)

# Real bash traceback text, session fbef66a33c18 style: a JSONDecodeError
# raised deep inside urllib, surfaced as stderr in the tool's error text.
_JSON_DECODE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 12, in <module>\n'
    "    data = json.loads(raw)\n"
    '  File ".../json/__init__.py", line 346, in loads\n'
    "    return _default_decoder.decode(s)\n"
    '  File ".../json/decoder.py", line 337, in decode\n'
    "    obj, end = self.raw_decode(s, idx=_w(s, 0).end())\n"
    "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
)

_FILE_NOT_FOUND_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 3, in <module>\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/x.json'"
)


def _failed(name: str, error: str) -> dict:
    return {"name": name, "ok": False, "error": error}


def _ok(name: str) -> dict:
    return {"name": name, "ok": True, "error": ""}


class TestErrorSignature:
    def test_variant_args_same_error_class_same_signature(self) -> None:
        r1 = _failed("bash", _JSON_DECODE_TRACEBACK)
        r2 = _failed(
            "bash",
            _JSON_DECODE_TRACEBACK.replace("line 12", "line 47"),
        )
        assert error_signature(r1) == error_signature(r2)

    def test_different_error_class_different_signature(self) -> None:
        r1 = _failed("bash", _JSON_DECODE_TRACEBACK)
        r2 = _failed("bash", _FILE_NOT_FOUND_TRACEBACK)
        assert error_signature(r1) != error_signature(r2)

    def test_signature_includes_tool_name(self) -> None:
        r1 = _failed("bash", _JSON_DECODE_TRACEBACK)
        r2 = _failed("other_tool", _JSON_DECODE_TRACEBACK)
        assert error_signature(r1) != error_signature(r2)

    def test_extracts_exact_exception_token(self) -> None:
        r = _failed("bash", _JSON_DECODE_TRACEBACK)
        assert error_signature(r) == "bash|json.decoder.JSONDecodeError"

    def test_fallback_bucket_when_no_exception_token(self) -> None:
        r1 = _failed("web_fetch", "connection refused on attempt 1")
        r2 = _failed("web_fetch", "connection refused on attempt 2")
        # digits stripped -> same bucket despite differing attempt numbers
        assert error_signature(r1) == error_signature(r2)

    def test_empty_error_text(self) -> None:
        r = _failed("bash", "")
        assert error_signature(r) == "bash|"


class TestTrailingStreak:
    def test_empty_records_returns_none(self) -> None:
        assert trailing_streak([]) is None

    def test_trailing_success_returns_none(self) -> None:
        records = [_failed("bash", _JSON_DECODE_TRACEBACK), _ok("bash")]
        assert trailing_streak(records) is None

    def test_fail_fail_succeed_stays_below_threshold(self) -> None:
        records = [
            _failed("bash", _JSON_DECODE_TRACEBACK),
            _failed("bash", _JSON_DECODE_TRACEBACK),
            _ok("bash"),
        ]
        assert trailing_streak(records) is None

    def test_alternating_error_classes_never_streak(self) -> None:
        records = [
            _failed("bash", _JSON_DECODE_TRACEBACK),
            _failed("bash", _FILE_NOT_FOUND_TRACEBACK),
            _failed("bash", _JSON_DECODE_TRACEBACK),
        ]
        result = trailing_streak(records)
        assert result is not None
        _, length, _ = result
        assert length == 1

    def test_third_same_signature_failure_crosses_threshold(self) -> None:
        records = [_failed("bash", _JSON_DECODE_TRACEBACK) for _ in range(3)]
        result = trailing_streak(records)
        assert result is not None
        signature, length, indexes = result
        assert length == STREAK_THRESHOLD
        assert signature == "bash|json.decoder.JSONDecodeError"
        assert indexes == [0, 1, 2]

    def test_streak_grows_past_threshold(self) -> None:
        records = [_failed("bash", _JSON_DECODE_TRACEBACK) for _ in range(8)]
        _, length, indexes = trailing_streak(records)
        assert length == 8
        assert indexes == list(range(8))

    @pytest.mark.parametrize("threshold_len", [1, 2])
    def test_below_threshold_lengths_reported_accurately(
        self, threshold_len: int
    ) -> None:
        records = [
            _failed("bash", _JSON_DECODE_TRACEBACK) for _ in range(threshold_len)
        ]
        _, length, _ = trailing_streak(records)
        assert length == threshold_len

"""Tests for pipeline/supersession.py: Cases A/B/C, stub text, and the gate."""

from __future__ import annotations

import pytest

from my_coding_agent.engine.tool_execution.schema import (
    EXTRACTION_INCOMPLETE_MARKER,
    SUPERSESSION_SIZE_FLOOR_CHARS,
)
from my_coding_agent.pipeline.supersession import (
    CASE_CONTAINMENT,
    CASE_IDENTICAL_CALL,
    CASE_INCOMPLETE_EXTRACT,
    STUB_PREFIX,
    build_stub,
    find_retirements,
    supersession_enabled,
)

_BIG = "x" * SUPERSESSION_SIZE_FLOOR_CHARS  # exactly at the floor
_BIGGER = _BIG + "y"  # strictly larger, still contains _BIG


def _record(name: str, tool_call_id: str, args: dict, ok: bool = True) -> dict:
    return {"name": name, "tool_call_id": tool_call_id, "args": args, "ok": ok}


def _tool_message(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class TestCaseAIncompleteExtract:
    def test_incomplete_extract_retired_after_later_successful_read(self) -> None:
        marker_text = f"partial stuff {EXTRACTION_INCOMPLETE_MARKER} cut.]" + _BIG
        records = [
            _record(
                "read_tool_artifact",
                "call_1",
                {"tool_call_id": "artifact_x", "query": "first query"},
            ),
            _record(
                "read_tool_artifact",
                "call_2",
                {"tool_call_id": "artifact_x", "query": "second query"},
            ),
        ]
        messages = [
            _tool_message("call_1", marker_text),
            _tool_message("call_2", _BIGGER),
        ]
        retirements = find_retirements(records, messages)
        assert len(retirements) == 1
        r = retirements[0]
        assert r.tool_call_id == "call_1"
        assert r.case == CASE_INCOMPLETE_EXTRACT
        assert r.superseding_tool_call_id == "call_2"

    def test_no_later_read_of_same_artifact_kept(self) -> None:
        marker_text = f"partial {EXTRACTION_INCOMPLETE_MARKER} cut.]" + _BIG
        records = [
            _record("read_tool_artifact", "call_1", {"tool_call_id": "artifact_x"}),
            _record("read_tool_artifact", "call_2", {"tool_call_id": "artifact_other"}),
        ]
        messages = [
            _tool_message("call_1", marker_text),
            _tool_message("call_2", _BIGGER),
        ]
        assert find_retirements(records, messages) == []

    def test_later_read_of_same_artifact_but_failed_kept(self) -> None:
        marker_text = f"partial {EXTRACTION_INCOMPLETE_MARKER} cut.]" + _BIG
        records = [
            _record("read_tool_artifact", "call_1", {"tool_call_id": "artifact_x"}),
            _record(
                "read_tool_artifact",
                "call_2",
                {"tool_call_id": "artifact_x"},
                ok=False,
            ),
        ]
        messages = [
            _tool_message("call_1", marker_text),
            _tool_message("call_2", _BIGGER),
        ]
        assert find_retirements(records, messages) == []

    def test_extract_without_marker_kept(self) -> None:
        records = [
            _record(
                "read_tool_artifact",
                "call_1",
                {"tool_call_id": "artifact_x", "query": "first query"},
            ),
            _record(
                "read_tool_artifact",
                "call_2",
                {"tool_call_id": "artifact_x", "query": "second query"},
            ),
        ]
        messages = [
            _tool_message("call_1", _BIG),  # no incompleteness marker
            _tool_message("call_2", "z" * len(_BIGGER)),  # no containment either
        ]
        assert find_retirements(records, messages) == []


class TestCaseBContainment:
    def test_contained_result_retired(self) -> None:
        records = [
            _record("bash", "call_1", {"cmd": "a"}),
            _record("bash", "call_2", {"cmd": "b"}),
        ]
        messages = [
            _tool_message("call_1", _BIG),
            _tool_message("call_2", "prefix-" + _BIG + "-suffix"),
        ]
        retirements = find_retirements(records, messages)
        assert len(retirements) == 1
        assert retirements[0].tool_call_id == "call_1"
        assert retirements[0].case == CASE_CONTAINMENT
        assert retirements[0].superseding_tool_call_id == "call_2"

    def test_fragment_only_containment_not_retired(self) -> None:
        # Only part of the earlier result's text occurs in the later one.
        records = [
            _record("bash", "call_1", {"cmd": "a"}),
            _record("bash", "call_2", {"cmd": "b"}),
        ]
        messages = [
            _tool_message("call_1", _BIG + "-tail-unique"),
            _tool_message("call_2", _BIG + "-different-tail"),
        ]
        assert find_retirements(records, messages) == []

    def test_semantic_overlap_without_containment_not_retired(self) -> None:
        records = [
            _record("bash", "call_1", {"cmd": "a"}),
            _record("bash", "call_2", {"cmd": "b"}),
        ]
        messages = [
            _tool_message("call_1", "A" * len(_BIG)),
            _tool_message("call_2", "B" * len(_BIG)),
        ]
        assert find_retirements(records, messages) == []

    def test_failed_newest_result_does_not_supersede(self) -> None:
        records = [
            _record("bash", "call_1", {"cmd": "a"}),
            _record("bash", "call_2", {"cmd": "b"}, ok=False),
        ]
        messages = [
            _tool_message("call_1", _BIG),
            _tool_message("call_2", "prefix-" + _BIG + "-suffix"),
        ]
        assert find_retirements(records, messages) == []


class TestCaseCIdenticalCall:
    def test_older_invocation_retired_when_newest_succeeds(self) -> None:
        args = {"path": "/tmp/x"}
        records = [
            _record("read_file", "call_1", args),
            _record("read_file", "call_2", args),
        ]
        messages = [
            _tool_message("call_1", _BIG),
            _tool_message("call_2", _BIG),
        ]
        retirements = find_retirements(records, messages)
        assert len(retirements) == 1
        assert retirements[0].tool_call_id == "call_1"
        assert retirements[0].case == CASE_IDENTICAL_CALL
        assert retirements[0].superseding_tool_call_id == "call_2"

    def test_newest_invocation_failed_nothing_retired(self) -> None:
        args = {"path": "/tmp/x"}
        records = [
            _record("read_file", "call_1", args),
            _record("read_file", "call_2", args, ok=False),
        ]
        messages = [
            _tool_message("call_1", _BIG),
            _tool_message("call_2", _BIG),
        ]
        assert find_retirements(records, messages) == []

    def test_different_args_not_retired(self) -> None:
        records = [
            _record("read_file", "call_1", {"path": "/tmp/x"}),
            _record("read_file", "call_2", {"path": "/tmp/y"}),
        ]
        messages = [
            _tool_message("call_1", "x" * len(_BIG)),
            _tool_message("call_2", "y" * len(_BIG)),
        ]
        assert find_retirements(records, messages) == []


class TestGate:
    def test_sub_floor_result_never_retired(self) -> None:
        small = "x" * (SUPERSESSION_SIZE_FLOOR_CHARS - 1)
        args = {"path": "/tmp/x"}
        records = [
            _record("read_file", "call_1", args),
            _record("read_file", "call_2", args),
        ]
        messages = [
            _tool_message("call_1", small),
            _tool_message("call_2", small),
        ]
        assert find_retirements(records, messages) == []

    def test_already_stubbed_message_not_reretired(self) -> None:
        args = {"path": "/tmp/x"}
        records = [
            _record("read_file", "call_1", args),
            _record("read_file", "call_2", args),
        ]
        stub_text = STUB_PREFIX + " already retired earlier" + _BIG
        messages = [
            _tool_message("call_1", stub_text),
            _tool_message("call_2", _BIG),
        ]
        assert find_retirements(records, messages) == []

    def test_supersession_enabled_default_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MCA_SUPERSESSION", raising=False)
        assert supersession_enabled() is True

    def test_supersession_disabled_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCA_SUPERSESSION", "0")
        assert supersession_enabled() is False

    def test_supersession_enabled_for_non_zero_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCA_SUPERSESSION", "1")
        assert supersession_enabled() is True


class TestBuildStub:
    def test_stub_names_tool_call_id_and_superseder(self) -> None:
        retirements = find_retirements(
            [
                _record("bash", "call_1", {"cmd": "a"}),
                _record("bash", "call_2", {"cmd": "b"}),
            ],
            [
                _tool_message("call_1", _BIG),
                _tool_message("call_2", "prefix-" + _BIG + "-suffix"),
            ],
        )
        stub = build_stub(retirements[0], artifact_path=None)
        assert stub.startswith(STUB_PREFIX)
        assert "call_1" in stub
        assert "call_2" in stub
        assert "read_tool_artifact" in stub
        assert "\n" not in stub  # one line

    def test_stub_includes_artifact_path_when_present(self) -> None:
        retirements = find_retirements(
            [
                _record("bash", "call_1", {"cmd": "a"}),
                _record("bash", "call_2", {"cmd": "b"}),
            ],
            [
                _tool_message("call_1", _BIG),
                _tool_message("call_2", "prefix-" + _BIG + "-suffix"),
            ],
        )
        stub = build_stub(
            retirements[0], artifact_path="/tmp/session/artifacts/call_1.stdout.txt"
        )
        assert "/tmp/session/artifacts/call_1.stdout.txt" in stub

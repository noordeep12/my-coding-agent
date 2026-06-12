"""Tests for the pure, network-free methods of LLM.

These cover argument aliasing/stripping, raw tool-call parsing, and the
zero-cost (phase-1) paths of route_tools. The LLM instance is built without
__init__ (see the bare_llm fixture) so no HTTP call to /models occurs.
"""

import httpx
import pytest

from my_coding_agent.llm import _BASELINE_TOOLS, _HTTP_RETRIES

# --- helpers -----------------------------------------------------------------


def _tool_def(name, tags=None):
    return {"function": {"name": name}, "tags": tags or []}


# --- _apply_arg_aliases ------------------------------------------------------


def test_apply_arg_aliases_bash_path_to_command(bare_llm):
    out = bare_llm._apply_arg_aliases("bash", {"path": "ls"})
    assert out == {"command": "ls"}


def test_apply_arg_aliases_does_not_override_existing(bare_llm):
    out = bare_llm._apply_arg_aliases("bash", {"path": "ls", "command": "pwd"})
    # command already present — alias must not clobber it.
    assert out["command"] == "pwd"


def test_apply_arg_aliases_unknown_tool_unchanged(bare_llm):
    args = {"foo": "bar"}
    assert bare_llm._apply_arg_aliases("read_article", dict(args)) == args


def test_apply_arg_aliases_read_file_variants(bare_llm):
    assert bare_llm._apply_arg_aliases("read_file", {"filename": "x"}) == {
        "file_path": "x"
    }


# --- _strip_unknown_args -----------------------------------------------------


def test_strip_unknown_args_drops_hallucinated_kwargs(bare_llm):
    out = bare_llm._strip_unknown_args("bash", {"command": "ls", "file_path": "x"})
    assert out == {"command": "ls"}


def test_strip_unknown_args_keeps_valid(bare_llm):
    out = bare_llm._strip_unknown_args("write_file", {"file_path": "p", "content": "c"})
    assert out == {"file_path": "p", "content": "c"}


def test_strip_unknown_args_unknown_func_returns_unchanged(bare_llm):
    args = {"whatever": 1}
    assert bare_llm._strip_unknown_args("does_not_exist", dict(args)) == args


# --- parse_tool_call ---------------------------------------------------------


def test_parse_tool_call_valid(bare_llm):
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    tid, name, args, err = bare_llm.parse_tool_call(tc)
    assert (tid, name, args, err) == ("call_1", "bash", {"command": "ls"}, None)


def test_parse_tool_call_missing_type(bare_llm):
    tc = {"id": "c2", "function": {"name": "bash"}}
    tid, name, args, err = bare_llm.parse_tool_call(tc)
    assert name is None
    assert "missing 'type'" in err


def test_parse_tool_call_wrong_type(bare_llm):
    tc = {"id": "c3", "type": "retrieval", "function": {"name": "bash"}}
    _, name, _, err = bare_llm.parse_tool_call(tc)
    assert name is None
    assert "not supported" in err


def test_parse_tool_call_missing_function_name(bare_llm):
    tc = {"id": "c4", "type": "function", "function": {}}
    _, name, _, err = bare_llm.parse_tool_call(tc)
    assert name is None
    assert "function.name" in err


def test_parse_tool_call_malformed_json_args(bare_llm):
    tc = {
        "id": "c5",
        "type": "function",
        "function": {"name": "bash", "arguments": "{not json}"},
    }
    tid, name, args, err = bare_llm.parse_tool_call(tc)
    assert name == "bash"  # name preserved for record creation
    assert args is None
    assert "could not parse" in err


def test_parse_tool_call_missing_id_uses_unknown(bare_llm):
    tc = {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
    tid, _, _, _ = bare_llm.parse_tool_call(tc)
    assert tid == "unknown_id"


# --- _request_with_retry -----------------------------------------------------


class _FlakySession:
    """Fails with a transient error for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise httpx.ConnectError("down")
        return "OK"


def test_request_with_retry_succeeds_after_transient_failures(bare_llm, mocker):
    mocker.patch("my_coding_agent.llm.time.sleep")  # don't actually wait
    bare_llm.session = _FlakySession(fail_times=_HTTP_RETRIES - 1)
    assert bare_llm._request_with_retry("GET", "http://x/models") == "OK"
    assert bare_llm.session.calls == _HTTP_RETRIES


def test_request_with_retry_raises_after_exhausting_attempts(bare_llm, mocker):
    mocker.patch("my_coding_agent.llm.time.sleep")
    bare_llm.session = _FlakySession(fail_times=_HTTP_RETRIES)
    with pytest.raises(httpx.ConnectError):
        bare_llm._request_with_retry("GET", "http://x/models")
    assert bare_llm.session.calls == _HTTP_RETRIES


def test_request_with_retry_does_not_retry_non_transient(bare_llm, mocker):
    sleep = mocker.patch("my_coding_agent.llm.time.sleep")

    class _ProtocolErrorSession:
        def request(self, method, url, **kwargs):
            raise httpx.HTTPStatusError(
                "nope", request=mocker.Mock(), response=mocker.Mock()
            )

    bare_llm.session = _ProtocolErrorSession()
    with pytest.raises(httpx.HTTPStatusError):
        bare_llm._request_with_retry("GET", "http://x/models")
    sleep.assert_not_called()


# --- route_tools (phase-1, no LLM call) --------------------------------------


def test_route_tools_empty_list_returns_empty(bare_llm):
    assert bare_llm.route_tools("anything", []) == []


def test_route_tools_only_baseline_returns_all(bare_llm):
    tools = [_tool_def(n) for n in _BASELINE_TOOLS]
    assert bare_llm.route_tools("hello", tools) == tools


def test_route_tools_keyword_match_selects_baseline_plus_matched(bare_llm):
    tools = [
        _tool_def("bash"),
        _tool_def("read_file"),
        _tool_def("read_tool_artifact"),
        _tool_def("read_article", tags=["web", "url"]),
    ]
    selected = bare_llm.route_tools("please fetch this url", tools)
    names = {t["function"]["name"] for t in selected}
    assert "read_article" in names
    assert _BASELINE_TOOLS <= names


def test_route_tools_baseline_tag_match_skips_phase2(bare_llm):
    tools = [
        _tool_def("bash", tags=["shell", "run"]),
        _tool_def("read_file"),
        _tool_def("read_tool_artifact"),
        _tool_def("read_article", tags=["web"]),
    ]
    # "run" matches a baseline tag but no non-baseline tag -> returns all tools,
    # never reaching the phase-2 LLM call.
    selected = bare_llm.route_tools("run the tests", tools)
    assert selected == tools

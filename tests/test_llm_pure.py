"""Tests for the pure, network-free collaborator and client methods.

These cover argument aliasing/stripping and raw tool-call parsing (on the
``ToolExecutor`` via the ``bare_executor`` fixture), the zero-cost (phase-1)
paths of ``ToolRouter.route_tools`` (via ``bare_router``), and the client's
``_request_with_retry`` (on ``bare_llm``). All fixtures are network-free.
"""

import httpx
import pytest

from my_coding_agent.engine.llm import _HTTP_RETRIES
from my_coding_agent.engine.routing import _BASELINE_TOOLS
from my_coding_agent.engine.tool_execution import args as arg_prep

# --- helpers -----------------------------------------------------------------


def _tool_def(name, tags=None):
    return {"function": {"name": name}, "tags": tags or []}


# --- apply_arg_aliases -------------------------------------------------------


def test_apply_arg_aliases_bash_path_to_command():
    out = arg_prep.apply_arg_aliases("bash", {"path": "ls"})
    assert out == {"command": "ls"}


def test_apply_arg_aliases_does_not_override_existing():
    out = arg_prep.apply_arg_aliases("bash", {"path": "ls", "command": "pwd"})
    # command already present — alias must not clobber it.
    assert out["command"] == "pwd"


def test_apply_arg_aliases_unknown_tool_unchanged():
    args = {"foo": "bar"}
    assert arg_prep.apply_arg_aliases("read_article", dict(args)) == args


def test_apply_arg_aliases_read_file_variants():
    assert arg_prep.apply_arg_aliases("read_file", {"filename": "x"}) == {
        "file_path": "x"
    }


# --- strip_unknown_args ------------------------------------------------------


def test_strip_unknown_args_drops_hallucinated_kwargs():
    out = arg_prep.strip_unknown_args("bash", {"command": "ls", "file_path": "x"})
    assert out == {"command": "ls"}


def test_strip_unknown_args_keeps_valid():
    out = arg_prep.strip_unknown_args("write_file", {"file_path": "p", "content": "c"})
    assert out == {"file_path": "p", "content": "c"}


def test_strip_unknown_args_unknown_func_returns_unchanged():
    args = {"whatever": 1}
    assert arg_prep.strip_unknown_args("does_not_exist", dict(args)) == args


# --- parse_tool_call ---------------------------------------------------------


def test_parse_tool_call_valid():
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    tid, name, args, err = arg_prep.parse_tool_call(tc)
    assert (tid, name, args, err) == ("call_1", "bash", {"command": "ls"}, None)


def test_parse_tool_call_missing_type():
    tc = {"id": "c2", "function": {"name": "bash"}}
    tid, name, args, err = arg_prep.parse_tool_call(tc)
    assert name is None
    assert "missing 'type'" in err


def test_parse_tool_call_wrong_type():
    tc = {"id": "c3", "type": "retrieval", "function": {"name": "bash"}}
    _, name, _, err = arg_prep.parse_tool_call(tc)
    assert name is None
    assert "not supported" in err


def test_parse_tool_call_missing_function_name():
    tc = {"id": "c4", "type": "function", "function": {}}
    _, name, _, err = arg_prep.parse_tool_call(tc)
    assert name is None
    assert "function.name" in err


def test_parse_tool_call_malformed_json_args():
    tc = {
        "id": "c5",
        "type": "function",
        "function": {"name": "bash", "arguments": "{not json}"},
    }
    tid, name, args, err = arg_prep.parse_tool_call(tc)
    assert name == "bash"  # name preserved for record creation
    assert args is None
    assert "could not parse" in err


def test_parse_tool_call_missing_id_uses_unknown():
    tc = {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
    tid, _, _, _ = arg_prep.parse_tool_call(tc)
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
    mocker.patch("my_coding_agent.engine.llm.time.sleep")  # don't actually wait
    bare_llm.session = _FlakySession(fail_times=_HTTP_RETRIES - 1)
    assert bare_llm._request_with_retry("GET", "http://x/models") == "OK"
    assert bare_llm.session.calls == _HTTP_RETRIES


def test_request_with_retry_raises_after_exhausting_attempts(bare_llm, mocker):
    mocker.patch("my_coding_agent.engine.llm.time.sleep")
    bare_llm.session = _FlakySession(fail_times=_HTTP_RETRIES)
    with pytest.raises(httpx.ConnectError):
        bare_llm._request_with_retry("GET", "http://x/models")
    assert bare_llm.session.calls == _HTTP_RETRIES


def test_request_with_retry_does_not_retry_non_transient(bare_llm, mocker):
    sleep = mocker.patch("my_coding_agent.engine.llm.time.sleep")

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


def test_route_tools_empty_list_returns_empty(bare_router):
    selected, phase = bare_router.route_tools("anything", [])
    assert selected == []
    assert phase == "empty"


def test_route_tools_only_baseline_returns_all(bare_router):
    tools = [_tool_def(n) for n in _BASELINE_TOOLS]
    selected, phase = bare_router.route_tools("hello", tools)
    assert selected == tools
    assert phase == "no_nonbaseline"


def test_route_tools_keyword_match_selects_baseline_plus_matched(bare_router):
    tools = [
        _tool_def("bash"),
        _tool_def("read_file"),
        _tool_def("read_tool_artifact"),
        _tool_def("read_article", tags=["web", "url"]),
    ]
    selected, phase = bare_router.route_tools("please fetch this url", tools)
    names = {t["function"]["name"] for t in selected}
    assert "read_article" in names
    assert _BASELINE_TOOLS <= names
    assert phase == "phase1_keyword"


def test_route_tools_baseline_tag_match_skips_phase2(bare_router):
    tools = [
        _tool_def("bash", tags=["shell", "run"]),
        _tool_def("read_file"),
        _tool_def("read_tool_artifact"),
        _tool_def("read_article", tags=["web"]),
    ]
    # "run" matches a baseline tag but no non-baseline tag -> returns all tools,
    # never reaching the phase-2 LLM call.
    selected, phase = bare_router.route_tools("run the tests", tools)
    assert selected == tools
    assert phase == "phase1_baseline"


# --- word-boundary tag matching -----------------------------------------------


def test_route_tools_substring_inside_word_does_not_match(bare_router, mocker):
    # No tag match anywhere sends routing to the phase-2 LLM call, so stub it to
    # return no tools -- otherwise a missing LLM server (e.g. CI) makes the call
    # fail and fall back to *all* tools, leaking read_article regardless of the
    # word-boundary behaviour under test.
    chat = mocker.patch.object(bare_router.client, "chat_completion")
    chat.return_value = mocker.Mock(
        json=lambda: {"choices": [{"message": {"role": "assistant", "content": "[]"}}]}
    )
    tools = [_tool_def("read_article", tags=["file"])]
    selected, phase = bare_router.route_tools("check the profile", tools)
    names = {t["function"]["name"] for t in selected}
    assert "read_article" not in names
    # No match anywhere, cold start (no previous selection) -> phase-2 fallback.
    assert phase == "phase2_llm"


def test_route_tools_whole_word_matches(bare_router):
    tools = [_tool_def("read_article", tags=["file"])]
    selected, phase = bare_router.route_tools("write the file", tools)
    names = {t["function"]["name"] for t in selected}
    assert "read_article" in names
    assert phase == "phase1_keyword"


# --- cold-start vs mid-run phase-2 gating -------------------------------------


def test_route_tools_mid_run_no_match_makes_no_llm_call(bare_router, mocker):
    chat = mocker.patch.object(bare_router.client, "chat_completion")
    tools = [_tool_def("read_article", tags=["web"])]
    selected, phase = bare_router.route_tools(
        "unrelated text", tools, has_previous_selection=True
    )
    assert selected is None
    assert phase == "carry_forward"
    chat.assert_not_called()


def test_route_tools_first_routing_no_match_uses_phase2(bare_router, mocker):
    chat = mocker.patch.object(bare_router.client, "chat_completion")
    chat.return_value = mocker.Mock(
        json=lambda: {"choices": [{"message": {"role": "assistant", "content": "[]"}}]}
    )
    tools = [_tool_def("read_article", tags=["web"])]
    selected, phase = bare_router.route_tools(
        "unrelated text", tools, has_previous_selection=False
    )
    assert phase == "phase2_llm"
    chat.assert_called_once()

"""Tests for the network-touching and dispatch logic of LLM.

Every test keeps the suite network-free (CONTRIBUTE.md §30): the HTTP layer is
mocked at ``_request_with_retry`` or ``chat_completion``, never reaching a real
server. These cover ``available_models``, ``chat_completion`` accounting,
``route_tools`` phase-2, output validation/summarization, tool dispatch, the
arg-correction loop, and the full ``execute_tool_calls`` flow (§42: behavior is
asserted, not merely executed).
"""

import json

import pytest

from my_coding_agent.llm import MAX_TOOL_OUTPUT_CHARS
from my_coding_agent.tools import ToolsRegistry


class _Resp:
    """Minimal httpx.Response stand-in returning a fixed JSON payload."""

    def __init__(self, payload, content=b"{}", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code
        self.text = content.decode() if isinstance(content, bytes) else str(content)

    def json(self):
        return self._payload


# --- setup_session -----------------------------------------------------------


def test_setup_session_applies_auth_and_timeout(bare_llm):
    bare_llm.api_key = "secret-token"
    bare_llm.timeout = 12.5
    bare_llm.setup_session()
    assert bare_llm.session.headers["Authorization"] == "Bearer secret-token"
    assert bare_llm.session.headers["content-type"] == "application/json"
    assert bare_llm.session.timeout.read == 12.5  # httpx wraps it in a Timeout


# --- available_models --------------------------------------------------------


def test_available_models_sets_context_window_from_match(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "my-model"
    payload = {
        "data": [
            {"id": "other", "context_length": 999},
            {"id": "my-model", "context_length": 50_000},
        ]
    }
    mocker.patch.object(bare_llm, "_request_with_retry", return_value=_Resp(payload))
    models = bare_llm.available_models()
    assert models == ["other", "my-model"]
    assert bare_llm.context_window == 50_000


def test_available_models_falls_back_when_model_absent(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "missing"
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"data": [{"id": "other", "context_length": 1}]}),
    )
    bare_llm.available_models()
    assert bare_llm.context_window == 131_072  # documented fallback


def test_available_models_uses_alternate_context_keys(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"data": [{"id": "m", "max_context_length": 4096}]}),
    )
    bare_llm.available_models()
    assert bare_llm.context_window == 4096


# --- lazy context_window (G-09) ---------------------------------------------


def test_construction_makes_no_http_request(mocker):
    """Regression (G-09): constructing LLM probes nothing — no /models call."""
    from my_coding_agent.llm import LLM

    probe = mocker.patch.object(LLM, "available_models")
    request = mocker.patch.object(LLM, "_request_with_retry")
    LLM()
    probe.assert_not_called()
    request.assert_not_called()


def test_context_window_resolves_lazily_on_first_use(bare_llm, mocker):
    """The /models probe fires on first context_window read, then is cached."""
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    request = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"data": [{"id": "m", "context_length": 7777}]}),
    )
    assert request.call_count == 0  # untouched until first access
    assert bare_llm.context_window == 7777
    assert request.call_count == 1
    assert bare_llm.context_window == 7777  # cached — no second probe
    assert request.call_count == 1


def test_context_window_falls_back_when_probe_fails(bare_llm, mocker):
    """An unreachable server yields the documented fallback, not an exception."""
    import httpx

    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    mocker.patch.object(
        bare_llm, "_request_with_retry", side_effect=httpx.ConnectError("down")
    )
    assert bare_llm.context_window == 131_072  # fallback, no raise


# --- chat_completion ---------------------------------------------------------


def test_chat_completion_records_usage(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    payload = {
        "choices": [{"message": {"content": "hi", "reasoning_content": "think"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    mocker.patch.object(bare_llm, "_request_with_retry", return_value=_Resp(payload))
    resp = bare_llm.chat_completion([{"role": "user", "content": "q"}], kind="main")
    assert resp.json() == payload
    assert bare_llm.llm_calls == [
        {"call": 1, "kind": "main", "prompt": 10, "completion": 4, "total": 14}
    ]


def test_chat_completion_passes_max_tokens(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    req = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"choices": [], "usage": {}}),
    )
    bare_llm.chat_completion([{"role": "user", "content": "q"}], max_tokens=128)
    assert req.call_args.kwargs["json"]["max_tokens"] == 128


def test_chat_completion_omits_max_tokens_when_none(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    req = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"choices": [], "usage": {}}),
    )
    bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert "max_tokens" not in req.call_args.kwargs["json"]


def test_chat_completion_non_json_body_raises_value_error(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"

    class _BadResp(_Resp):
        def json(self):
            raise ValueError("not json")

    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_BadResp(None, content=b"<html>", status_code=500),
    )
    with pytest.raises(ValueError, match="non-JSON response"):
        bare_llm.chat_completion([{"role": "user", "content": "q"}])


# --- route_tools phase-2 (LLM fallback) --------------------------------------


def _tool(name, tags=None):
    return {"function": {"name": name}, "tags": tags or []}


def test_route_tools_phase2_selects_llm_choice_plus_baseline(bare_llm, mocker):
    tools = [
        _tool("bash"),
        _tool("read_file"),
        _tool("read_tool_artifact"),
        _tool("read_article", tags=["web"]),
    ]
    # Message matches no tag anywhere → phase-2 LLM fallback runs.
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": '["read_article"]'}}]}),
    )
    selected = {t["function"]["name"] for t in bare_llm.route_tools("xyzzy", tools)}
    assert "read_article" in selected
    assert {"bash", "read_file", "read_tool_artifact"} <= selected


def test_route_tools_phase2_falls_back_to_all_on_bad_json(bare_llm, mocker):
    tools = [
        _tool("bash"),
        _tool("read_file"),
        _tool("read_tool_artifact"),
        _tool("read_article", tags=["web"]),
    ]
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "not an array"}}]}),
    )
    selected = bare_llm.route_tools("xyzzy", tools)
    assert selected == tools  # unparseable → keep all tools


def test_route_tools_phase2_extracts_array_from_prose(bare_llm, mocker):
    tools = [
        _tool("bash"),
        _tool("read_file"),
        _tool("read_tool_artifact"),
        _tool("read_article", tags=["web"]),
    ]
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'Sure! Here you go: ["read_article"] done.'
                        }
                    }
                ]
            }
        ),
    )
    selected = {t["function"]["name"] for t in bare_llm.route_tools("xyzzy", tools)}
    assert "read_article" in selected


# --- _validate_tool_output ---------------------------------------------------


def test_validate_tool_output_empty_returns_placeholder(bare_llm):
    bare_llm._session_log_path = None
    out = bare_llm._validate_tool_output("   ", "bash")
    assert out == "(tool returned empty output)"


def test_validate_tool_output_truncates_oversized(bare_llm):
    bare_llm._session_log_path = "/tmp/log"
    big = "y" * (MAX_TOOL_OUTPUT_CHARS + 50)
    out = bare_llm._validate_tool_output(big, "read_file")
    assert "[output truncated" in out
    assert "/tmp/log" in out


def test_validate_tool_output_summary_not_truncated(bare_llm):
    bare_llm._session_log_path = None
    big = "z" * (MAX_TOOL_OUTPUT_CHARS + 50)
    assert bare_llm._validate_tool_output(big, "bash", is_summary=True) == big


# --- _summarize_artifact -----------------------------------------------------


def test_summarize_artifact_uses_llm_summary(bare_llm, mocker):
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "all good"}}]}),
    )
    out = bare_llm._summarize_artifact({"exit_code": 0, "ok": True}, "bash", "call_1")
    assert out.startswith("all good")
    assert 'read_tool_artifact(tool_call_id="call_1")' in out


def test_summarize_artifact_falls_back_on_llm_failure_bash(bare_llm, mocker):
    mocker.patch.object(bare_llm, "chat_completion", side_effect=RuntimeError("boom"))
    out = bare_llm._summarize_artifact(
        {"exit_code": 2, "ok": False, "stdout": "abc", "stderr": ""}, "bash", "c1"
    )
    head = json.loads(out.split("\n[Full output")[0])
    assert head == {"exit_code": 2, "ok": False, "stdout_chars": 3, "stderr_chars": 0}


def test_summarize_artifact_falls_back_on_llm_failure_file(bare_llm, mocker):
    mocker.patch.object(bare_llm, "chat_completion", side_effect=RuntimeError("boom"))
    out = bare_llm._summarize_artifact(
        {"content": "...", "file_path": "/a.txt", "size": 99}, "read_file", "c2"
    )
    head = json.loads(out.split("\n[Full output")[0])
    assert head == {"file_path": "/a.txt", "size": 99}


# --- _dispatch_tool ----------------------------------------------------------


def test_dispatch_tool_plain_string_result(bare_llm, tmp_path):
    bare_llm._session_log_path = None
    f = tmp_path / "f.txt"
    f.write_text("data")
    reg = ToolsRegistry(base_dir=str(tmp_path))
    result, is_artifact, is_truncated = bare_llm._dispatch_tool(
        reg, "read_file", {"file_path": str(f)}, "c1"
    )
    assert result == "data"
    assert is_artifact is False
    assert is_truncated is False


def test_dispatch_tool_artifact_tuple_is_summarized(bare_llm, mocker):
    bare_llm._session_log_path = None
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"content": "summary"}}]}),
    )

    class _Reg:
        def big(self):
            return None, {"exit_code": 0, "ok": True, "stdout": "x", "stderr": ""}

    result, is_artifact, _ = bare_llm._dispatch_tool(_Reg(), "big", {}, "c1")
    assert is_artifact is True
    assert "summary" in result
    assert bare_llm.tool_artifacts["c1"]["exit_code"] == 0


# --- invoke_tool -------------------------------------------------------------


def test_invoke_tool_success(bare_llm, tmp_path):
    bare_llm._session_log_path = None
    f = tmp_path / "f.txt"
    f.write_text("hello")
    reg = ToolsRegistry(base_dir=str(tmp_path))
    result, status, record = bare_llm.invoke_tool(
        "c1", "read_file", {"file_path": str(f)}, reg, {}
    )
    assert result == "hello"
    assert status == "success"
    assert record["ok"] is True


def test_invoke_tool_unknown_tool_returns_error(bare_llm):
    reg = ToolsRegistry()
    result, status, record = bare_llm.invoke_tool("c1", "does_not_exist", {}, reg, {})
    assert status == "error"
    assert "not found" in result
    assert record["ok"] is False


def test_invoke_tool_recoverable_exception_returns_error(bare_llm):
    """read_file traversal raises PathTraversalError (a ValueError), recoverable."""
    bare_llm._session_log_path = None
    reg = ToolsRegistry()  # base_dir = cwd
    result, status, record = bare_llm.invoke_tool(
        "c1", "read_file", {"file_path": "../../../etc/passwd"}, reg, {}
    )
    assert status == "error"
    assert "PathTraversalError" in result
    assert record["ok"] is False


def test_invoke_tool_non_recoverable_exception_reraises(bare_llm):
    """A registry whose tool raises a non-recoverable error (KeyError) re-raises."""
    bare_llm._session_log_path = None

    class _Reg(ToolsRegistry):
        def read_file(self, file_path: str) -> str:  # same signature as the real tool
            raise KeyError("fatal")

    with pytest.raises(KeyError):
        bare_llm.invoke_tool("c1", "read_file", {"file_path": "x"}, _Reg(), {})


def test_invoke_tool_corrects_wrong_args_then_succeeds(bare_llm, mocker):
    """First dispatch raises TypeError; correction fixes args; the retry succeeds."""
    bare_llm._session_log_path = None
    bare_llm.messages = []
    bare_llm.tools = []
    calls = {"n": 0}

    class _Reg(ToolsRegistry):
        def read_file(self, file_path: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise TypeError("unexpected keyword argument 'wrong'")
            return f"read {file_path}"

    mocker.patch.object(bare_llm, "_correct_args", return_value={"file_path": "ok"})
    result, status, _ = bare_llm.invoke_tool(
        "c1", "read_file", {"wrong": "x"}, _Reg(), {}
    )
    assert status == "success"
    assert result == "read ok"


# --- execute_tool_calls (full flow) ------------------------------------------


def test_execute_tool_calls_success_flow(bare_llm):
    """A valid read_file call inside the workspace dispatches and records success."""
    bare_llm._session_log_path = None
    bare_llm.tools = []
    bare_llm.tool_artifacts = {}
    bare_llm._before_hook = lambda name, args: args
    bare_llm._after_hook = lambda name, args, result: result
    # Read a file that exists under the cwd (the registry's default base_dir).
    message = {
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"file_path": "pyproject.toml"}),
                },
            }
        ]
    }
    messages, records = bare_llm.execute_tool_calls(message)
    assert len(messages) == 1
    assert records[0]["name"] == "read_file"
    assert records[0]["ok"] is True
    assert "[project]" in messages[0]["content"]


def test_execute_tool_calls_skips_on_before_hook_none(bare_llm):
    bare_llm._session_log_path = None
    bare_llm.tools = []
    bare_llm.tool_artifacts = {}
    bare_llm._before_hook = lambda name, args: None  # skip every call
    bare_llm._after_hook = lambda name, args, result: result
    message = {
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ]
    }
    messages, records = bare_llm.execute_tool_calls(message)
    assert messages[0]["status"] == "skipped"
    assert records[0]["status"] == "skipped"


def test_execute_tool_calls_parse_error_recorded(bare_llm):
    bare_llm._session_log_path = None
    bare_llm.tools = []
    bare_llm.tool_artifacts = {}
    message = {"tool_calls": [{"id": "c1", "function": {"name": "bash"}}]}  # no type
    messages, records = bare_llm.execute_tool_calls(message)
    assert messages[0]["status"] == "error"
    assert records[0]["ok"] is False


def test_execute_tool_calls_empty_returns_empty(bare_llm):
    bare_llm._session_log_path = None
    bare_llm.tools = []
    bare_llm.tool_artifacts = {}
    messages, records = bare_llm.execute_tool_calls({"tool_calls": []})
    assert messages == []
    assert records == []


# --- _correct_args -----------------------------------------------------------


def test_correct_args_returns_parsed_corrected_call(bare_llm, mocker):
    bare_llm.messages = []
    bare_llm.tools = []
    fixed = {
        "function": {"name": "read_file", "arguments": json.dumps({"file_path": "/ok"})}
    }
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"tool_calls": [fixed]}}]}),
    )
    out = bare_llm._correct_args(
        "read_file", {"bad": 1}, TypeError("boom"), "(sig)", {}, "c1", attempt=0
    )
    assert out == {"file_path": "/ok"}


def test_correct_args_returns_none_when_model_skips_tool(bare_llm, mocker):
    bare_llm.messages = []
    bare_llm.tools = []
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"tool_calls": []}}]}),
    )
    out = bare_llm._correct_args(
        "read_file", {}, TypeError("boom"), "(sig)", {}, "c1", attempt=0
    )
    assert out is None


def test_correct_args_returns_none_on_unparseable_args(bare_llm, mocker):
    bare_llm.messages = []
    bare_llm.tools = []
    bad_call = {"function": {"name": "read_file", "arguments": "{not json}"}}
    mocker.patch.object(
        bare_llm,
        "chat_completion",
        return_value=_Resp({"choices": [{"message": {"tool_calls": [bad_call]}}]}),
    )
    out = bare_llm._correct_args(
        "read_file", {}, TypeError("boom"), "(sig)", {}, "c1", attempt=0
    )
    assert out is None


# --- _validate_tool_output bash non-JSON warning -----------------------------


def test_validate_tool_output_warns_on_non_json_bash(bare_llm):
    bare_llm._session_log_path = None
    # Non-JSON bash output passes through unchanged (the warning is a side effect).
    out = bare_llm._validate_tool_output("plain text not json", "bash")
    assert out == "plain text not json"


# --- before_tool_call / after_tool_call --------------------------------------


def test_before_tool_call_applies_alias_and_hook(bare_llm):
    bare_llm._before_hook = lambda name, args: args
    out = bare_llm.before_tool_call("c1", "bash", {"path": "ls"})
    assert out == {"command": "ls"}  # alias path→command applied


def test_before_tool_call_hook_can_skip(bare_llm):
    bare_llm._before_hook = lambda name, args: None
    assert bare_llm.before_tool_call("c1", "bash", {"command": "ls"}) is None


def test_after_tool_call_applies_hook(bare_llm):
    bare_llm._after_hook = lambda name, args, result: result.upper()
    assert bare_llm.after_tool_call("c1", "bash", {}, "ok") == "OK"


def test_after_tool_call_swallows_hook_exception(bare_llm):
    def _raises(name, args, result):
        raise RuntimeError("hook broke")

    bare_llm._after_hook = _raises
    # On hook failure the original result is returned unchanged.
    assert bare_llm.after_tool_call("c1", "bash", {}, "original") == "original"

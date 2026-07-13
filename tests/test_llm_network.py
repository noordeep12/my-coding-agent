"""Tests for the network-touching client and dispatch-collaborator logic.

Every test keeps the suite network-free (CONTRIBUTE.md §30): the HTTP layer is
mocked at ``_request_with_retry`` or ``chat_completion``, never reaching a real
server. These cover the client (``available_models``, ``chat_completion``
accounting), and the executor's output validation/summarization, tool dispatch,
arg-correction loop, and full ``execute_tool_calls`` flow (``ToolExecutor`` via
``bare_executor``) — §42: behavior is asserted, not merely executed.
"""

import json

import pytest

from my_coding_agent.engine.tool_execution import MAX_TOOL_OUTPUT_CHARS, output
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry


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
    bare_llm.api_key = "secret-token"  # pragma: allowlist secret
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
    from my_coding_agent.engine.llm import LLM

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


def test_chat_completion_captures_tools_on_event(bare_llm, mocker, tmp_path):
    from my_coding_agent.observability.recorder import Recorder

    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    bare_llm._context_window = 8192  # avoid the lazy network probe
    sdir = tmp_path / "sid"
    sdir.mkdir()
    bare_llm._recorder = Recorder(session_id="sid", session_dir=sdir)
    tools = [{"type": "function", "function": {"name": "bash"}}]
    req = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"choices": [{"message": {"content": "ok"}}], "usage": {}}),
    )
    bare_llm.chat_completion(
        [{"role": "user", "content": "q"}], tools=tools, kind="main"
    )
    # The request body still carries the tools (capture must not alter it).
    assert req.call_args.kwargs["json"]["tools"] == tools
    # And the recorded event carries the same tool definitions.
    events = [
        json.loads(line)
        for line in (sdir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert events[-1]["tools"] == tools


def test_chat_completion_passes_max_tokens(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    req = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"choices": [{"message": {"content": "ok"}}], "usage": {}}),
    )
    bare_llm.chat_completion([{"role": "user", "content": "q"}], max_tokens=128)
    assert req.call_args.kwargs["json"]["max_tokens"] == 128


def test_chat_completion_omits_max_tokens_when_none(bare_llm, mocker):
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"
    req = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"choices": [{"message": {"content": "ok"}}], "usage": {}}),
    )
    bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert "max_tokens" not in req.call_args.kwargs["json"]


def test_chat_completion_non_json_body_raises_value_error(bare_llm, mocker):
    # A 2xx body that fails to parse as JSON classifies as malformed-body (a
    # ValueError via APIResponseError), immediately — status is 2xx so this is
    # not an http-status failure.
    bare_llm.api_url = "http://x/v1"
    bare_llm.model = "m"

    class _BadResp(_Resp):
        def json(self):
            raise ValueError("not json")

    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_BadResp(None, content=b"<html>", status_code=200),
    )
    with pytest.raises(ValueError, match="non-JSON response"):
        bare_llm.chat_completion([{"role": "user", "content": "q"}])


# --- validate_tool_output ----------------------------------------------------


def test_validate_tool_output_empty_returns_placeholder():
    out = output.validate_tool_output("   ", "bash", None)
    assert out == "(tool returned empty output)"


def test_validate_tool_output_truncates_oversized():
    big = "y" * (MAX_TOOL_OUTPUT_CHARS + 50)
    out = output.validate_tool_output(big, "read_file", "/tmp/log")
    assert "[output truncated" in out
    assert "/tmp/log" in out


def test_validate_tool_output_summary_not_truncated():
    big = "z" * (MAX_TOOL_OUTPUT_CHARS + 50)
    assert output.validate_tool_output(big, "bash", None, is_summary=True) == big


# --- read_tool_artifact extraction (via the executor's injected LLM) --------


def test_read_tool_artifact_extraction_uses_llm(bare_executor, mocker):
    mocker.patch.object(
        bare_executor.llm,
        "chat_completion",
        return_value=_Resp(
            {"choices": [{"message": {"content": "the relevant line"}}]}
        ),
    )
    reg = ToolsRegistry(artifacts={"call_1": "full stored text"}, llm=bare_executor.llm)
    out = reg.read_tool_artifact("call_1", "find the relevant line")
    assert out == "the relevant line"


def test_read_tool_artifact_extraction_falls_back_on_llm_failure(bare_executor, mocker):
    mocker.patch.object(
        bare_executor.llm, "chat_completion", side_effect=RuntimeError("boom")
    )
    reg = ToolsRegistry(
        artifacts={"call_1": {"stdout": "abc", "stderr": ""}}, llm=bare_executor.llm
    )
    out = reg.read_tool_artifact("call_1", "anything")
    assert "abc" in out
    assert "Extraction unavailable" in out


def _invoke(executor, tool_call_id, func_name, call_args, registry):
    """Drive invoke_tool → after_tool_call and return (env, status, record).

    invoke_tool returns only (raw, failure); the canonical envelope is produced
    by after_tool_call, so the final result needs both phases. The registry is
    injected on the executor (instance attribute).
    """
    executor.registry = registry
    raw, failure = executor.invoke_tool(tool_call_id, func_name, call_args)
    content, status, record = executor.after_tool_call(
        tool_call_id, func_name, call_args, raw, failure
    )
    return json.loads(content), status, record


# --- invoke_tool dispatch (post-processed: plain output + artifact tuple) ----


def test_invoke_tool_plain_string_not_truncated(bare_executor, tmp_path):
    bare_executor.llm._session_log_path = None
    f = tmp_path / "f.txt"
    f.write_text("data")
    reg = ToolsRegistry(base_dir=str(tmp_path))
    env, status, record = _invoke(
        bare_executor, "c1", "read_file", {"file_path": str(f)}, reg
    )
    assert status == "success"
    assert env["output"] == "data"
    assert record["truncated"] is False


def test_invoke_tool_artifact_tuple_is_previewed(bare_executor, mocker):
    """A (None, dict) artifact tuple with a large stdout is offloaded: the agent
    gets a bounded preview excerpt + skim guidance (not the whole blob).
    """
    bare_executor.llm._session_log_path = None
    from my_coding_agent.engine.tool_execution.output import PREVIEW_MAX_CHARS

    body = "x" * (PREVIEW_MAX_CHARS * 2) + "TAILZZZ"
    mocker.patch.object(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: (
            None,
            {"exit_code": 0, "ok": True, "stdout": body, "stderr": ""},
        ),
    )
    env, status, record = _invoke(
        bare_executor, "c1", "bash", {"command": "x"}, ToolsRegistry()
    )
    assert "[Preview:" in env["output"]
    assert "TAILZZZ" not in env["output"]  # bounded, not the whole blob
    assert env["metadata"]["preview"]["stdout"]["total_bytes"] == len(body)
    assert record["artifact"] is True
    assert bare_executor.tool_artifacts["c1"]["exit_code"] == 0


# --- invoke_tool -------------------------------------------------------------


def test_invoke_tool_success(bare_executor, tmp_path):
    bare_executor.llm._session_log_path = None
    f = tmp_path / "f.txt"
    f.write_text("hello")
    reg = ToolsRegistry(base_dir=str(tmp_path))
    env, status, record = _invoke(
        bare_executor, "c1", "read_file", {"file_path": str(f)}, reg
    )
    assert env["ok"] is True
    assert env["output"] == "hello"
    assert env["tool"] == "read_file"
    assert status == "success"
    assert record["ok"] is True


def test_invoke_tool_bash_success_envelope(bare_executor):
    """Bash structured JSON maps into the schema with ok / exit_code metadata."""
    bare_executor.llm._session_log_path = None

    class _Reg(ToolsRegistry):
        def bash(self, command: str, timeout: int = 60):
            return json.dumps(
                {"stdout": "hi", "stderr": "", "exit_code": 0, "ok": True}
            )

    env, status, record = _invoke(
        bare_executor, "c1", "bash", {"command": "echo hi"}, _Reg()
    )
    assert env["ok"] is True
    assert env["output"] == "hi"
    assert env["metadata"]["exit_code"] == 0
    assert status == "success"
    assert record["ok"] is True


def test_invoke_tool_bash_failure_envelope(bare_executor):
    """A non-zero exit (returned as data, not raised) becomes a schema failure."""
    bare_executor.llm._session_log_path = None

    class _Reg(ToolsRegistry):
        def bash(self, command: str, timeout: int = 60):
            return json.dumps(
                {"stdout": "", "stderr": "boom", "exit_code": 1, "ok": False}
            )

    env, status, record = _invoke(
        bare_executor, "c1", "bash", {"command": "false"}, _Reg()
    )
    assert env["ok"] is False
    assert env["error"] == "boom"
    assert env["metadata"]["exit_code"] == 1
    assert status == "error"
    assert record["ok"] is False


def test_invoke_tool_error_string_becomes_failure(bare_executor, tmp_path):
    """A tool that returns an 'Error…' string (no exception) is flagged failure."""
    bare_executor.llm._session_log_path = None
    reg = ToolsRegistry(base_dir=str(tmp_path))
    env, status, _ = _invoke(
        bare_executor, "c1", "read_file", {"file_path": str(tmp_path / "nope.txt")}, reg
    )
    assert env["ok"] is False
    assert "not found" in env["error"]
    assert status == "error"


def test_invoke_tool_unknown_tool_returns_error(bare_executor):
    env, status, record = _invoke(
        bare_executor, "c1", "does_not_exist", {}, ToolsRegistry()
    )
    assert status == "error"
    assert env["ok"] is False
    assert "not found" in env["error"]
    assert record["ok"] is False


def test_invoke_tool_recoverable_exception_returns_error(bare_executor):
    """read_file traversal raises PathTraversalError (a ValueError), recoverable."""
    bare_executor.llm._session_log_path = None
    env, status, record = _invoke(
        bare_executor,
        "c1",
        "read_file",
        {"file_path": "../../../etc/passwd"},
        ToolsRegistry(),  # base_dir = cwd
    )
    assert status == "error"
    assert env["ok"] is False
    assert "PathTraversalError" in env["error"]
    assert record["ok"] is False


def test_invoke_tool_non_recoverable_exception_reraises(bare_executor):
    """A registry whose tool raises a non-recoverable error (KeyError) re-raises."""
    bare_executor.llm._session_log_path = None

    class _Reg(ToolsRegistry):
        def read_file(self, file_path: str) -> str:  # same signature as the real tool
            raise KeyError("fatal")

    bare_executor.registry = _Reg()
    with pytest.raises(KeyError):
        bare_executor.invoke_tool("c1", "read_file", {"file_path": "x"})


def test_invoke_tool_wrong_args_fails_without_correction(bare_executor):
    """A TypeError (wrong args) fails directly — no LLM correction, no retry."""
    bare_executor.llm._session_log_path = None

    class _Reg(ToolsRegistry):
        def read_file(self, file_path: str) -> str:
            raise TypeError("unexpected keyword argument 'wrong'")

    env, status, _ = _invoke(
        bare_executor, "c1", "read_file", {"file_path": "x"}, _Reg()
    )
    assert status == "error"
    assert env["metadata"]["reason"] == "wrong_args"


# --- run (full flow) ---------------------------------------------------------


def test_run_success_flow(bare_executor):
    """A valid read_file call inside the workspace dispatches and records success."""
    bare_executor.llm._session_log_path = None
    bare_executor.tool_calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"file_path": "pyproject.toml"}),
            },
        }
    ]
    messages, records = bare_executor.run()
    assert len(messages) == 1
    assert records[0]["name"] == "read_file"
    assert records[0]["ok"] is True
    envelope = json.loads(messages[0]["content"])
    assert envelope["ok"] is True
    assert envelope["tool"] == "read_file"
    assert "[project]" in envelope["output"]


def test_run_parse_error_recorded(bare_executor):
    bare_executor.llm._session_log_path = None
    bare_executor.tool_calls = [{"id": "c1", "function": {"name": "bash"}}]  # no type
    messages, records = bare_executor.run()
    assert messages[0]["status"] == "error"
    assert records[0]["ok"] is False


def test_run_empty_returns_empty(bare_executor):
    messages, records = bare_executor.run()
    assert messages == []
    assert records == []


# --- validate_tool_output bash non-JSON warning ------------------------------


def test_validate_tool_output_warns_on_non_json_bash():
    # Non-JSON bash output passes through unchanged (the warning is a side effect).
    out = output.validate_tool_output("plain text not json", "bash", None)
    assert out == "plain text not json"


# --- before_tool_call --------------------------------------------------------


def test_before_tool_call_applies_alias(bare_executor):
    out = bare_executor.before_tool_call("bash", {"path": "ls"})
    assert out == {"command": "ls"}  # alias path->command applied

"""Tests for tool-execution helpers (summary extraction + output schema).

The block tagged "characterization" locks the CURRENT observable behavior of
``ToolExecutor`` before the module is split into a package. These tests are the
safety net for that refactor: they assert behavior as it is today (including the
``error``/``metadata.stderr`` duplication), so any accidental change during the
move is caught. They must stay green across the refactor unchanged.
"""

import json

import pytest

from my_coding_agent.engine.tool_execution import (
    MAX_TOOL_OUTPUT_CHARS,
    TOOL_SCHEMA_VERSION,
    ToolExecutor,
    _extract_summary,
    args,
    build_tool_result,
    output,
    validate_tool_result,
)
from my_coding_agent.engine.tool_execution.schema import result_envelope
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry


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


# ── characterization: envelope normalization (_result_envelope) ───────────────
# Locks how raw tool returns become the canonical envelope, per source shape.


def test_envelope_bash_success_folds_stdout_and_metadata():
    raw = json.dumps({"stdout": "hi", "stderr": "", "exit_code": 0, "ok": True})
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is True
    assert env["output"] == "hi"
    assert env["error"] is None
    assert env["metadata"]["exit_code"] == 0


def test_envelope_bash_failure_duplicates_stderr_into_error_and_metadata():
    # CURRENT behavior (locked, not endorsed): stderr appears in BOTH error and
    # metadata.stderr. Issue #55 will dedup this; until then the net pins it.
    raw = json.dumps({"stdout": "", "stderr": "boom", "exit_code": 1, "ok": False})
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is False
    assert env["error"] == "boom"
    assert env["metadata"]["stderr"] == "boom"
    assert env["metadata"]["exit_code"] == 1


def test_envelope_error_string_convention_marks_failure():
    env = result_envelope("read_file", "Error: nope", False, False, "c1")
    assert env["ok"] is False
    assert env["error"] == "Error: nope"
    assert env["output"] == ""


def test_envelope_plain_string_is_success():
    env = result_envelope("read_file", "done", False, False, "c1")
    assert env["ok"] is True
    assert env["output"] == "done"
    assert env["error"] is None


def test_envelope_truncated_flag_is_recorded():
    env = result_envelope("read_file", "done", False, True, "c1")
    assert env["metadata"]["truncated"] is True


def test_envelope_artifact_branch_reads_stored_artifact():
    artifact = {"stdout": "x", "stderr": "err", "exit_code": 1, "ok": False}
    env = result_envelope("bash", "summary text", True, False, "c1", artifact)
    assert env["ok"] is False
    assert env["output"] == "summary text"
    assert env["error"] == "err"
    assert env["metadata"]["summarized"] is True
    assert env["metadata"]["artifact"] is True


# ── characterization: tool-call parsing (parse_tool_call) ─────────────────────


def test_parse_tool_call_valid():
    tc = {
        "id": "c1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
    }
    assert args.parse_tool_call(tc) == ("c1", "bash", {"command": "ls"}, None)


@pytest.mark.parametrize(
    "tc,needle",
    [
        ({"id": "c1", "function": {"name": "bash"}}, "missing 'type'"),
        (
            {"id": "c1", "type": "web", "function": {"name": "bash"}},
            "not supported",
        ),
        ({"id": "c1", "type": "function", "function": {}}, "missing 'function.name'"),
        (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bash", "arguments": "{bad"},
            },
            "could not parse",
        ),
    ],
)
def test_parse_tool_call_errors(tc, needle):
    _id, _name, _args, error = args.parse_tool_call(tc)
    assert error is not None and needle in error


# ── characterization: argument preparation ────────────────────────────────────


def test_apply_arg_aliases_remaps_known_wrong_name():
    assert args.apply_arg_aliases("bash", {"path": "ls"}) == {"command": "ls"}


def test_strip_unknown_args_drops_kwargs_not_in_signature():
    cleaned = args.strip_unknown_args("bash", {"command": "ls", "bogus": 1})
    assert cleaned == {"command": "ls"}


# ── characterization: output validation (validate_tool_output) ────────────────


def test_validate_output_replaces_empty():
    assert output.validate_tool_output("   ", "bash", None) == (
        "(tool returned empty output)"
    )


def test_validate_output_truncates_oversized():
    long = "x" * (MAX_TOOL_OUTPUT_CHARS * 2)
    out = output.validate_tool_output(long, "read_file", None)
    assert "[output truncated" in out
    assert len(out) < len(long)


# ── characterization: dispatch + retry + exceptions (invoke_tool) ─────────────


def _registry(executor, base_dir=None):
    # base_dir lets file tools resolve inside a tmp dir instead of the repo root,
    # so tests neither pollute the workspace nor trip the path-traversal guard.
    return ToolsRegistry(artifacts=executor.tool_artifacts, tools=[], base_dir=base_dir)


def _invoke(executor, tool_call_id, func_name, call_args, registry):
    """Drive invoke_tool → after_tool_call and return (env, status, record).

    The two-phase contract (invoke = call, after = post-process) means the final
    envelope only exists once both have run; this stitches them. The registry is
    injected on the executor (instance attribute).
    """
    executor.registry = registry
    raw, failure = executor.invoke_tool(tool_call_id, func_name, call_args)
    content, status, record = executor.after_tool_call(
        tool_call_id, func_name, call_args, raw, failure
    )
    return json.loads(content), status, record


def test_invoke_tool_success(bare_executor, tmp_path):
    target = tmp_path / "out.txt"
    env, status, record = _invoke(
        bare_executor,
        "c1",
        "write_file",
        {"file_path": "out.txt", "content": "hi"},
        _registry(bare_executor, str(tmp_path)),
    )
    assert status == "success"
    assert env["ok"] is True
    assert record["ok"] is True
    assert target.read_text() == "hi"


def test_invoke_tool_not_found(bare_executor):
    env, status, _ = _invoke(bare_executor, "c1", "nope", {}, _registry(bare_executor))
    assert status == "error"
    assert env["ok"] is False
    assert env["metadata"]["reason"] == "not_found"


def test_invoke_tool_error_string_is_failure(bare_executor, tmp_path):
    env, status, _ = _invoke(
        bare_executor,
        "c1",
        "read_file",
        {"file_path": "missing.txt"},
        _registry(bare_executor, str(tmp_path)),
    )
    assert status == "error"
    assert env["ok"] is False


def test_invoke_tool_recoverable_exception_returns_error(bare_executor, monkeypatch):
    def boom(self, file_path):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(ToolsRegistry, "read_file", boom)
    env, status, _ = _invoke(
        bare_executor, "c1", "read_file", {"file_path": "x"}, _registry(bare_executor)
    )
    assert status == "error"
    assert env["ok"] is False
    assert env["metadata"]["reason"] == "raised"


def test_invoke_tool_non_recoverable_exception_reraises(bare_executor, monkeypatch):
    def boom(self, file_path):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(ToolsRegistry, "read_file", boom)
    bare_executor.registry = _registry(bare_executor)
    with pytest.raises(RuntimeError, match="kaboom"):
        bare_executor.invoke_tool("c1", "read_file", {"file_path": "x"})


# ── characterization: orchestration (run) ─────────────────────────────────────


def _tool_call(name, args, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_run_success(bare_executor, monkeypatch):
    # The executor's registry is a real ToolsRegistry; stub bash so the
    # orchestration path is exercised without a real shell or network.
    monkeypatch.setattr(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: json.dumps(
            {"stdout": "hi", "stderr": "", "exit_code": 0, "ok": True}
        ),
    )
    bare_executor.tool_calls = [_tool_call("bash", {"command": "ls"})]
    messages, records = bare_executor.run()
    assert messages[0]["role"] == "tool"
    assert messages[0]["status"] == "success"
    assert records[0]["ok"] is True


def test_run_parse_error(bare_executor):
    bare_executor.tool_calls = [{"id": "c1", "function": {"name": "bash"}}]  # no type
    messages, records = bare_executor.run()
    assert messages[0]["status"] == "error"
    assert records[0]["ok"] is False


# --- toolset forwarding (regression: subagents were spawned with 0 tools) ----


def test_executor_forwards_tools_to_registry(bare_llm):
    """The registry must be built with the toolset passed to ToolExecutor.

    Regression for the zero-tools bug: the executor built ToolRegistry without
    forwarding ``tools``, so ``delegate`` saw an empty ``_tools`` and spawned
    subagents with no tools at all.
    """
    tools = [{"function": {"name": "bash"}}, {"function": {"name": "delegate"}}]
    executor = ToolExecutor({"tool_calls": []}, bare_llm, tools=tools)
    assert executor.registry._tools == tools


def test_executor_defaults_to_empty_toolset(bare_llm):
    executor = ToolExecutor({"tool_calls": []}, bare_llm)
    assert executor.registry._tools == []

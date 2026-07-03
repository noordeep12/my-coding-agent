"""Tests for tool-execution helpers (summary extraction + output schema).

The block tagged "characterization" locks the observable behavior of
``ToolExecutor`` — how raw tool returns become the canonical envelope. For a
bash result each datum maps to one field: stdout→output, stderr→error (whenever
non-empty, regardless of ok), exit_code→metadata, success→ok; stderr is never
duplicated into metadata.
"""

import json

import pytest

from my_coding_agent.engine.tool_execution import (
    MAX_TOOL_OUTPUT_CHARS,
    TOOL_SCHEMA_VERSION,
    ToolExecutor,
    args,
    build_tool_result,
    output,
    validate_tool_result,
)
from my_coding_agent.engine.tool_execution.envelope import result_envelope
from my_coding_agent.engine.tool_execution.output import (
    PREVIEW_MAX_CHARS,
    build_stream_preview,
)
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.observability import current_session_id

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


def test_envelope_bash_failure_puts_stderr_in_error_not_metadata():
    # Deduped: stderr lives ONLY in `error`, never copied into metadata.
    raw = json.dumps({"stdout": "", "stderr": "boom", "exit_code": 1, "ok": False})
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is False
    assert env["error"] == "boom"
    assert "stderr" not in env["metadata"]
    assert env["metadata"]["exit_code"] == 1


def test_envelope_bash_masked_failure_surfaces_stderr_with_ok_true():
    # exit 0 + empty stdout + stderr → ok:true with a non-null error (masked failure).
    raw = json.dumps(
        {"stdout": "", "stderr": "grep: bad -P", "exit_code": 0, "ok": True}
    )
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is True
    assert env["output"] == ""
    assert env["error"] == "grep: bad -P"
    assert "stderr" not in env["metadata"]


def test_envelope_bash_success_with_stderr_keeps_ok_true():
    # A successful command that prints to stderr (git/curl-style) stays ok:true.
    raw = json.dumps(
        {
            "stdout": "data",
            "stderr": "Switched to branch main",
            "exit_code": 0,
            "ok": True,
        }
    )
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is True
    assert env["output"] == "data"
    assert env["error"] == "Switched to branch main"


def test_envelope_bash_failure_empty_stderr_yields_null_error():
    # grep no-match style: non-zero exit, empty stderr → error null (no placeholder).
    raw = json.dumps({"stdout": "", "stderr": "", "exit_code": 1, "ok": False})
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is False
    assert env["error"] is None
    assert env["metadata"]["exit_code"] == 1
    assert "stderr" not in env["metadata"]


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


def test_envelope_artifact_branch_uses_composed_error_and_exit_code():
    # The executor composes stdout→result and stderr→error per stream; the artifact
    # branch attaches them verbatim (no stderr in metadata) and reads ok/exit_code.
    artifact = {"stdout": "x", "stderr": "err", "exit_code": 1, "ok": False}
    env = result_envelope(
        "bash",
        "stdout preview",
        True,
        False,
        "c1",
        artifact,
        preview={"stdout": {"shown": 1, "total": 1}},
        error="err",
    )
    assert env["ok"] is False
    assert env["output"] == "stdout preview"
    assert env["error"] == "err"
    assert env["metadata"]["exit_code"] == 1
    assert "stderr" not in env["metadata"]
    assert env["metadata"]["artifact"] is True


def test_envelope_structured_tuple_metadata_bag_merges_untouched():
    # A structured-return tool (e.g. read_article) can carry its own metadata
    # bag on the tuple's dict; it arrives in the envelope's metadata untouched,
    # on both the offloaded and small-body (no preview) paths.
    artifact = {
        "stdout": "body",
        "ok": True,
        "metadata": {"content_type": "application/json", "transform": "none"},
    }
    env = result_envelope("read_article", "body", True, False, "c1", artifact)
    assert env["metadata"]["content_type"] == "application/json"
    assert env["metadata"]["transform"] == "none"
    assert "artifact" not in env["metadata"]
    assert "tool_call_id" not in env["metadata"]
    assert "preview" not in env["metadata"]


def test_envelope_structured_tuple_ok_false_not_sniffed_as_bash():
    # A fetched JSON body containing an "ok" key must not be reinterpreted as
    # the bash structured contract when the tool isn't bash.
    env = result_envelope("read_article", '{"ok": false, "x": 1}', False, False, "c1")
    assert env["ok"] is True
    assert env["output"] == '{"ok": false, "x": 1}'


def test_envelope_bash_ok_sniff_still_applies_to_bash():
    raw = json.dumps({"ok": False, "stdout": "out", "stderr": "err", "exit_code": 1})
    env = result_envelope("bash", raw, False, False, "c1")
    assert env["ok"] is False
    assert env["output"] == "out"
    assert env["error"] == "err"


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


def test_executor_forwards_llm_to_registry(bare_llm):
    """The registry must receive the executor's LLM client (design D5) so
    read_tool_artifact can make its bounded extraction call."""
    executor = ToolExecutor({"tool_calls": []}, bare_llm)
    assert executor.registry._llm is bare_llm


# ── artifact preview: bounded excerpt + skim guidance in `output` ─────────────


def test_build_stream_preview_bounds_output_and_reports_true_totals():
    body = "HEAD\n" + ("x" * (PREVIEW_MAX_CHARS + 500)) + "\nTAILMARKER"
    output_text, preview = build_stream_preview(body, "/s/artifacts/c1.stdout.txt")
    # Bounded: the whole body (and its tail) is NOT in the agent-facing value.
    assert "TAILMARKER" not in output_text
    assert len(output_text) < len(body)
    # True totals reported; excerpt is smaller than the full body.
    assert preview["total_bytes"] == len(body)
    assert preview["shown_bytes"] < preview["total_bytes"]
    assert preview["full_output_path"] == "/s/artifacts/c1.stdout.txt"
    # Guidance is inline in the value, naming the path + skim tools.
    assert "[Preview:" in output_text
    assert "/s/artifacts/c1.stdout.txt" in output_text
    assert "grep" in output_text and "sed" in output_text
    assert "guidance" not in preview


def test_build_stream_preview_small_shown_in_full():
    output_text, preview = build_stream_preview("hello", None)
    assert output_text.startswith("hello")
    assert preview["shown_bytes"] == preview["total_bytes"] == 5


def test_executor_offloads_stdout_stream_and_omits_full_output(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """A large stdout is written to its per-stream file at creation, and the
    envelope carries only the bounded stdout preview under metadata.preview.stdout."""
    monkeypatch.chdir(tmp_path)
    body = "HEAD\n" + ("x" * (PREVIEW_MAX_CHARS + 500)) + "\nTAILMARKER"
    mocker.patch.object(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: (
            None,
            {"exit_code": 0, "ok": True, "stdout": body, "stderr": ""},
        ),
    )
    token = current_session_id.set("sess123")
    try:
        env, _status, _record = _invoke(
            bare_executor, "call1", "bash", {"command": "x"}, ToolsRegistry()
        )
    finally:
        current_session_id.reset(token)

    art = tmp_path / ".my_coding_agent" / "sess123" / "artifacts" / "call1.stdout.txt"
    assert art.exists()
    assert art.read_text() == body  # full stdout on disk
    assert "TAILMARKER" not in env["output"]  # full stdout NOT in the envelope
    assert env["metadata"]["preview"]["stdout"]["full_output_path"].endswith(
        "call1.stdout.txt"
    )
    assert env["metadata"]["preview"]["stdout"]["total_bytes"] == len(body)


def test_executor_offloads_large_stderr_into_error(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """A large stderr with empty stdout is bounded in `error` and written to its
    own .stderr.txt file — not flooded verbatim into the envelope."""
    monkeypatch.chdir(tmp_path)
    err = "ERRHEAD\n" + ("e" * (PREVIEW_MAX_CHARS + 500)) + "\nERRTAIL"
    mocker.patch.object(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: (
            None,
            {"exit_code": 1, "ok": False, "stdout": "", "stderr": err},
        ),
    )
    token = current_session_id.set("sess123")
    try:
        env, _status, _record = _invoke(
            bare_executor, "call1", "bash", {"command": "x"}, ToolsRegistry()
        )
    finally:
        current_session_id.reset(token)

    stderr_file = (
        tmp_path / ".my_coding_agent" / "sess123" / "artifacts" / "call1.stderr.txt"
    )
    stdout_file = (
        tmp_path / ".my_coding_agent" / "sess123" / "artifacts" / "call1.stdout.txt"
    )
    assert stderr_file.exists() and stderr_file.read_text() == err
    assert not stdout_file.exists()  # empty stdout writes no file
    assert env["ok"] is False
    assert env["output"] == "(tool returned empty output)"  # stdout was empty
    assert "ERRTAIL" not in env["error"]  # full stderr NOT in the envelope
    assert "[Preview:" in env["error"]
    assert env["metadata"]["preview"]["stderr"]["total_bytes"] == len(err)
    assert "stdout" not in env["metadata"]["preview"]


def test_executor_offloads_both_streams_separately(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """Both streams large → each previewed into its own file; preview has both keys."""
    monkeypatch.chdir(tmp_path)
    out = "OUT\n" + ("o" * (PREVIEW_MAX_CHARS + 500)) + "\nOUTTAIL"
    err = "ERR\n" + ("e" * (PREVIEW_MAX_CHARS + 500)) + "\nERRTAIL"
    mocker.patch.object(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: (
            None,
            {"exit_code": 2, "ok": False, "stdout": out, "stderr": err},
        ),
    )
    token = current_session_id.set("sess123")
    try:
        env, _status, _record = _invoke(
            bare_executor, "call1", "bash", {"command": "x"}, ToolsRegistry()
        )
    finally:
        current_session_id.reset(token)

    base = tmp_path / ".my_coding_agent" / "sess123" / "artifacts"
    assert (base / "call1.stdout.txt").read_text() == out
    assert (base / "call1.stderr.txt").read_text() == err
    assert "OUTTAIL" not in env["output"] and "[Preview:" in env["output"]
    assert "ERRTAIL" not in env["error"] and "[Preview:" in env["error"]
    assert set(env["metadata"]["preview"]) == {"stdout", "stderr"}
    assert env["metadata"]["exit_code"] == 2


def test_executor_offloads_large_json_fetch_with_disclosure(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """A large JSON read_article fetch offloads losslessly and the envelope
    discloses content_type/transform without mislabeling small bodies."""
    monkeypatch.chdir(tmp_path)
    payload = {"note": "line1\\nline2", "big": "x" * (PREVIEW_MAX_CHARS + 500)}
    body = json.dumps(payload)
    mocker.patch.object(
        ToolsRegistry,
        "read_article",
        staticmethod(
            lambda url, timeout=15.0: (
                None,
                {
                    "stdout": body,
                    "ok": True,
                    "metadata": {
                        "content_type": "application/json",
                        "transform": "none",
                    },
                },
            )
        ),
    )
    token = current_session_id.set("sess123")
    try:
        env, _status, _record = _invoke(
            bare_executor,
            "call1",
            "read_article",
            {"url": "https://example.com/data.json"},
            ToolsRegistry(),
        )
    finally:
        current_session_id.reset(token)

    art = tmp_path / ".my_coding_agent" / "sess123" / "artifacts" / "call1.stdout.txt"
    assert json.loads(art.read_text()) == payload
    assert env["metadata"]["content_type"] == "application/json"
    assert env["metadata"]["transform"] == "none"
    assert env["metadata"]["artifact"] is True
    assert env["ok"] is True


def test_executor_small_verbatim_fetch_not_labeled_as_offloaded(bare_executor, mocker):
    """A small structured-tuple return (no stream over the preview budget) must
    not carry metadata.artifact — it was never actually offloaded."""
    mocker.patch.object(
        ToolsRegistry,
        "read_article",
        staticmethod(
            lambda url, timeout=15.0: (
                None,
                {
                    "stdout": '{"ok": false}',
                    "ok": True,
                    "metadata": {
                        "content_type": "application/json",
                        "transform": "none",
                    },
                },
            )
        ),
    )
    env, _status, _record = _invoke(
        bare_executor,
        "call1",
        "read_article",
        {"url": "https://example.com/data.json"},
        ToolsRegistry(),
    )
    assert "artifact" not in env["metadata"]
    assert env["ok"] is True
    assert env["output"] == '{"ok": false}'


def test_executor_inlines_small_stderr_when_stdout_offloaded(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """When stdout is large but stderr is small, stderr stays inline in `error`."""
    monkeypatch.chdir(tmp_path)
    body = "x" * (PREVIEW_MAX_CHARS + 500)
    mocker.patch.object(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: (
            None,
            {"exit_code": 0, "ok": True, "stdout": body, "stderr": "just a warning"},
        ),
    )
    token = current_session_id.set("sess123")
    try:
        env, _status, _record = _invoke(
            bare_executor, "call1", "bash", {"command": "x"}, ToolsRegistry()
        )
    finally:
        current_session_id.reset(token)

    assert env["error"] == "just a warning"  # small stderr inline, no preview
    assert "stderr" not in env["metadata"]["preview"]
    assert "stdout" in env["metadata"]["preview"]
    assert not (
        tmp_path / ".my_coding_agent" / "sess123" / "artifacts" / "call1.stderr.txt"
    ).exists()


def test_write_artifact_file_returns_none_on_oserror(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """A failed artifact write (full disk / permissions) must not abort the run:
    ``_write_artifact_file`` returns None and the offload still yields a valid
    envelope with no on-disk copy (preview falls back to read_tool_artifact)."""
    monkeypatch.chdir(tmp_path)
    body = "HEAD\n" + ("x" * (PREVIEW_MAX_CHARS + 500)) + "\nTAILMARKER"
    mocker.patch.object(
        ToolsRegistry,
        "bash",
        lambda self, command, timeout=60: (
            None,
            {"exit_code": 0, "ok": True, "stdout": body, "stderr": ""},
        ),
    )

    def _boom(self, *_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)

    token = current_session_id.set("sess123")
    try:
        env, status, _record = _invoke(
            bare_executor, "call1", "bash", {"command": "x"}, ToolsRegistry()
        )
    finally:
        current_session_id.reset(token)

    art = tmp_path / ".my_coding_agent" / "sess123" / "artifacts" / "call1.stdout.txt"
    assert not art.exists()  # no on-disk copy was written
    assert status == "success"  # offload continued despite the write failure
    assert env["metadata"]["artifact"] is True
    assert env["metadata"]["preview"]["stdout"]["full_output_path"] is None


def test_executor_writes_artifacts_under_each_session_dir(
    bare_executor, tmp_path, monkeypatch, mocker
):
    """Each subagent writes to its own session directory (spec scenario).

    Subagent isolation rests on the ``current_session_id`` contextvar being set
    per (sub)agent run. Offloading under two different session ids must land each
    artifact under its own ``.my_coding_agent/<session>/artifacts/<id>.txt`` with
    its own contents, never leaking into the other's dir; and ``read_tool_artifact``
    under one session must not see the other's file. Covers the subagent-isolation
    edge case (§18/§30), not just the single-session happy path.
    """
    monkeypatch.chdir(tmp_path)
    filler = "x" * (PREVIEW_MAX_CHARS + 500)

    def fake_bash(self, command, timeout=60):
        body = f"BODY-{command}\n{filler}\n"
        return None, {"exit_code": 0, "ok": True, "stdout": body, "stderr": ""}

    mocker.patch.object(ToolsRegistry, "bash", fake_bash)

    def offload(session_id, call_id, command):
        token = current_session_id.set(session_id)
        try:
            _invoke(
                bare_executor, call_id, "bash", {"command": command}, ToolsRegistry()
            )
        finally:
            current_session_id.reset(token)

    offload("sessA", "callA", "A")
    offload("sessB", "callB", "B")

    artifacts = tmp_path / ".my_coding_agent"
    art_a = artifacts / "sessA" / "artifacts" / "callA.stdout.txt"
    art_b = artifacts / "sessB" / "artifacts" / "callB.stdout.txt"
    # Each run's full output landed under its own session dir with its own body.
    assert art_a.read_text() == f"BODY-A\n{filler}\n"
    assert art_b.read_text() == f"BODY-B\n{filler}\n"
    # Neither run leaked into the other session's artifacts dir.
    assert not (artifacts / "sessA" / "artifacts" / "callB.stdout.txt").exists()
    assert not (artifacts / "sessB" / "artifacts" / "callA.stdout.txt").exists()
    # read_tool_artifact scoped to sessA cannot retrieve sessB's file.
    token = current_session_id.set("sessA")
    try:
        out = ToolsRegistry().read_tool_artifact("callB", "anything")
        assert "no artifact found" in out
    finally:
        current_session_id.reset(token)

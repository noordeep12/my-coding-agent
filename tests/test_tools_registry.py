"""Tests for ToolsRegistry tool methods.

Filesystem is exercised via tmp_path (real but isolated and cheap). subprocess
and httpx are mocked so no real command runs and no real network call is made.
"""

import json
import subprocess

import httpx
import pytest

from my_coding_agent.engine.agent import DEFAULT_MAX_STEPS
from my_coding_agent.engine.tool_registry import ARTIFACT_THRESHOLD
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.observability import current_session_id

# --- read_file / write_file --------------------------------------------------


def test_write_then_read_round_trip(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    target = tmp_path / "sub" / "out.txt"
    msg = reg.write_file(str(target), "hello world")
    assert "Written 11 bytes" in msg
    assert target.read_text() == "hello world"


def test_read_file_missing_returns_error(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    out = reg.read_file(str(tmp_path / "no_such_file.xyz"))
    assert out.startswith("Error: file not found:")


def test_write_file_creates_parent_dirs(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    target = tmp_path / "a" / "b" / "c.txt"
    reg.write_file(str(target), "x")
    assert target.exists()


def test_read_file_reads_existing(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    f = tmp_path / "f.txt"
    f.write_text("content")
    assert reg.read_file(str(f)) == "content"


def test_read_file_relative_path_resolves_under_base(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    (tmp_path / "rel.txt").write_text("data")
    assert reg.read_file("rel.txt") == "data"


def test_read_file_large_file_returns_artifact_tuple(tmp_path):
    """A file over ARTIFACT_THRESHOLD is offloaded — not lossily truncated — so
    the full content is preserved for the artifact store."""
    reg = ToolsRegistry(base_dir=str(tmp_path))
    big = "x" * (ARTIFACT_THRESHOLD + 1)
    f = tmp_path / "big.txt"
    f.write_text(big)
    out = reg.read_file(str(f))
    assert isinstance(out, tuple)
    assert out[0] is None
    assert out[1]["ok"] is True
    assert out[1]["stdout"] == big


def test_read_file_small_file_stays_inline(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    f = tmp_path / "small.txt"
    f.write_text("x" * ARTIFACT_THRESHOLD)
    out = reg.read_file(str(f))
    assert isinstance(out, str)


def test_read_file_traversal_raises(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal detected"):
        reg.read_file("../../etc/passwd")


def test_write_file_absolute_outside_base_raises(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal detected"):
        reg.write_file("/tmp/escape.txt", "x")


# --- read_tool_artifact ------------------------------------------------------


def test_read_tool_artifact_requires_query():
    reg = ToolsRegistry(artifacts={"call_1": {"k": "v"}})
    out = reg.read_tool_artifact("call_1", "")
    assert "Error" in out and "query" in out and "read_tool_artifact(" in out


def test_read_tool_artifact_found_dict_no_llm_falls_back_to_head_excerpt():
    """No LLM client injected → bounded head excerpt, never the whole artifact."""
    reg = ToolsRegistry(artifacts={"call_1": {"stdout": "v" * 10, "stderr": ""}})
    out = reg.read_tool_artifact("call_1", "what does it say")
    assert "v" * 10 in out
    assert "Extraction unavailable" in out


def test_read_tool_artifact_found_str_no_llm_falls_back_to_head_excerpt():
    reg = ToolsRegistry(artifacts={"call_1": "raw text"})
    out = reg.read_tool_artifact("call_1", "what does it say")
    assert "raw text" in out
    assert "Extraction unavailable" in out


def test_read_tool_artifact_missing():
    reg = ToolsRegistry(artifacts={})
    out = reg.read_tool_artifact("nope", "anything")
    assert "no artifact found" in out


def test_read_tool_artifact_reads_session_file_cross_step(tmp_path, monkeypatch):
    """Regression for #64: a later step's registry has an empty in-memory store,
    yet retrieval still succeeds by reading the on-disk per-artifact file."""
    monkeypatch.chdir(tmp_path)
    art_dir = tmp_path / ".my_coding_agent" / "sessX" / "artifacts"
    art_dir.mkdir(parents=True)
    (art_dir / "call9.stdout.txt").write_text("FULL CONTENT")
    token = current_session_id.set("sessX")
    try:
        reg = ToolsRegistry(artifacts={})  # empty, like a later step's registry
        out = reg.read_tool_artifact("call9", "content")
        assert "FULL CONTENT" in out
    finally:
        current_session_id.reset(token)


def test_read_tool_artifact_missing_file_falls_back_to_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    token = current_session_id.set("sessX")
    try:
        reg = ToolsRegistry(artifacts={})
        assert "no artifact found" in reg.read_tool_artifact("absent", "anything")
    finally:
        current_session_id.reset(token)


def test_read_tool_artifact_rejects_unsafe_id(tmp_path, monkeypatch):
    """A crafted id cannot traverse out of the artifacts directory; it skips the
    file path and falls back to the in-memory store (miss → error)."""
    monkeypatch.chdir(tmp_path)
    token = current_session_id.set("sessX")
    try:
        reg = ToolsRegistry(artifacts={})
        out = reg.read_tool_artifact("../../etc/passwd", "anything")
        assert "no artifact found" in out
    finally:
        current_session_id.reset(token)


def test_read_tool_artifact_uses_llm_for_extraction(mocker):
    """With an LLM injected, read_tool_artifact makes a bounded extraction call
    tagged with the artifact_query kind instead of falling back to a head excerpt."""
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {"choices": [{"message": {"content": "the relevant passage"}}]}
    )
    reg = ToolsRegistry(artifacts={"call_1": "some stored text"}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "find the passage")
    assert out == "the relevant passage"
    _, kwargs = fake_llm.chat_completion.call_args
    assert kwargs["kind"] == "artifact_query"


def test_read_tool_artifact_absent_content_says_not_found(mocker):
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {"choices": [{"message": {"content": "NOT FOUND"}}]}
    )
    reg = ToolsRegistry(artifacts={"call_1": "some stored text"}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "something absent")
    assert "No content relevant" in out


def test_read_tool_artifact_extraction_failure_falls_back_to_head_excerpt(mocker):
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.side_effect = RuntimeError("server down")
    reg = ToolsRegistry(artifacts={"call_1": "some stored text"}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "anything")
    assert "some stored text" in out
    assert "Extraction unavailable" in out


def test_read_tool_artifact_huge_artifact_stays_within_budget(mocker):
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_CHUNK_MAX_CHARS,
        EXTRACTION_OUTPUT_MAX_CHARS,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [{"message": {"content": "x" * EXTRACTION_OUTPUT_MAX_CHARS}}]
        }
    )
    huge = "y" * (EXTRACTION_CHUNK_MAX_CHARS * 5)
    reg = ToolsRegistry(artifacts={"call_1": huge}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "detail near the end")
    assert len(out) <= EXTRACTION_OUTPUT_MAX_CHARS
    # Chunked scan stops calling once the output budget is filled, not after
    # every chunk (huge is 5 chunks; the first chunk already fills the budget).
    assert fake_llm.chat_completion.call_count == 1


def test_read_tool_artifact_reaches_detail_near_the_end_via_chunk_scan(mocker):
    """The relevant detail lives only in the last of several chunks; the scan
    must not stop at the first NOT FOUND — it reaches the whole stored output."""
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_CHUNK_MAX_CHARS,
    )

    huge = "y" * (EXTRACTION_CHUNK_MAX_CHARS * 3)  # 3 chunks

    def fake_chat_completion(messages, **kwargs):
        prompt = messages[0]["content"]
        is_last_chunk = prompt.rstrip().endswith("y")
        content = "THE DETAIL" if is_last_chunk else "NOT FOUND"
        payload = {"choices": [{"message": {"content": content}}]}
        return mocker.Mock(json=lambda: payload)

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.side_effect = fake_chat_completion
    reg = ToolsRegistry(artifacts={"call_1": huge}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "detail near the end")
    assert "THE DETAIL" in out
    assert fake_llm.chat_completion.call_count == 3


# --- bash --------------------------------------------------------------------


def test_bash_success_returns_json(mocker):
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="echo hi", returncode=0, stdout="hi\n", stderr=""
        ),
    )
    out = ToolsRegistry().bash("echo hi")
    parsed = json.loads(out)
    assert parsed == {"stdout": "hi", "stderr": "", "exit_code": 0, "ok": True}


def test_bash_nonzero_exit_sets_ok_false(mocker):
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="false", returncode=1, stdout="", stderr="boom"
        ),
    )
    parsed = json.loads(ToolsRegistry().bash("false"))
    assert parsed["ok"] is False
    assert parsed["exit_code"] == 1
    assert parsed["stderr"] == "boom"


def test_bash_timeout_returns_error_json(mocker):
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 99", timeout=60),
    )
    parsed = json.loads(ToolsRegistry().bash("sleep 99"))
    assert parsed["ok"] is False
    assert parsed["exit_code"] == -1
    assert "timed out" in parsed["stderr"]


def test_bash_large_output_returns_artifact_tuple(mocker):
    big = "x" * (ARTIFACT_THRESHOLD + 1)
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="cat big", returncode=0, stdout=big, stderr=""
        ),
    )
    out = ToolsRegistry().bash("cat big")
    assert isinstance(out, tuple)
    assert out[0] is None
    assert out[1]["ok"] is True
    assert len(out[1]["stdout"]) == ARTIFACT_THRESHOLD + 1


def test_bash_at_threshold_boundary_returns_json(mocker):
    """Exactly at threshold (not above) stays inline as JSON, not an artifact."""
    exact = "x" * ARTIFACT_THRESHOLD
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="cat", returncode=0, stdout=exact, stderr=""
        ),
    )
    out = ToolsRegistry().bash("cat")
    assert isinstance(out, str)


# --- read_article ------------------------------------------------------------


def test_read_article_converts_html(mocker):
    resp = mocker.Mock()
    resp.text = "<h1>Title</h1><p>Body</p>"
    resp.raise_for_status = mocker.Mock()
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert "Title" in out
    assert "Body" in out


def test_read_article_http_error(mocker):
    err_resp = mocker.Mock(status_code=404)
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get",
        side_effect=httpx.HTTPStatusError(
            "nf", request=mocker.Mock(), response=err_resp
        ),
    )
    out = ToolsRegistry.read_article("https://example.com/missing")
    assert "HTTP 404" in out


def test_read_article_generic_error(mocker):
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get",
        side_effect=httpx.ConnectError("down"),
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert out.startswith("Error fetching")


def test_read_article_large_page_returns_artifact_tuple(mocker):
    """A page whose converted markdown exceeds ARTIFACT_THRESHOLD is offloaded —
    not lossily truncated — so the full text stays in the artifact store."""
    resp = mocker.Mock()
    resp.text = "<p>" + ("word " * 20000) + "</p>"  # ~100_000 chars, well over
    # ARTIFACT_THRESHOLD (8_000) but under the fetch-side sanity cap.
    resp.raise_for_status = mocker.Mock()
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert isinstance(out, tuple)
    assert out[0] is None
    assert out[1]["ok"] is True
    assert "word" in out[1]["stdout"]
    assert "[...truncated" not in out[1]["stdout"]  # lossless: no pre-truncation


def test_read_article_fetch_sanity_cap_truncates_pathological_page(mocker):
    """A page far beyond the fetch-side sanity cap is truncated at the cap —
    the cap guards a pathological page, fidelity within it is still offloaded."""
    from my_coding_agent.engine.tool_registry.registry import ARTICLE_FETCH_MAX_CHARS

    resp = mocker.Mock()
    resp.text = "<p>" + ("word " * 100_000) + "</p>"  # far beyond the sanity cap
    resp.raise_for_status = mocker.Mock()
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert isinstance(out, tuple)
    assert "[...truncated" in out[1]["stdout"]
    assert len(out[1]["stdout"]) < ARTICLE_FETCH_MAX_CHARS + 200


# --- delegate ----------------------------------------------------------------


def _make_fake_agent(
    mocker,
    stop_reason="stop",
    final_text="final turn text",
    handback_report=None,
    report="report text",
):
    """Return a mock AgentNode in a given finish state."""
    fake = mocker.Mock()
    fake.session_id = "abc123"
    fake.stop_reason = stop_reason
    fake.final_assistant_text.return_value = final_text
    fake.handback_report = handback_report
    fake.generate_report.return_value = report
    return fake


def test_delegate_clean_finish_returns_final_turn_verbatim(mocker):
    """A clean stop hands back the final assistant turn with no synthesis."""
    fake_agent = _make_fake_agent(mocker, stop_reason="stop", final_text="final turn")
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    out = ToolsRegistry().delegate(task="do X", context="ctx")
    assert out == "final turn"
    fake_agent.execute.assert_called_once_with(max_steps=DEFAULT_MAX_STEPS)
    fake_agent.generate_report.assert_not_called()
    fake_agent.recorder.record_report.assert_called_once_with("final turn")


def test_delegate_cutoff_returns_pipeline_report(mocker):
    """A cutoff hands back the pipeline-synthesized report, never the last turn."""
    fake_agent = _make_fake_agent(
        mocker, stop_reason="max_steps", handback_report="synthesized report"
    )
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    out = ToolsRegistry().delegate(task="do X", context="ctx")
    assert out == "synthesized report"
    fake_agent.final_assistant_text.assert_not_called()
    fake_agent.generate_report.assert_not_called()
    fake_agent.recorder.record_report.assert_called_once_with("synthesized report")


def test_delegate_empty_final_turn_falls_back_to_generate_report(mocker):
    """A clean stop with no usable final text falls back to synthesis."""
    fake_agent = _make_fake_agent(
        mocker, stop_reason="stop", final_text="", report="fallback report"
    )
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    out = ToolsRegistry().delegate(task="do X", context="ctx")
    assert out == "fallback report"
    fake_agent.generate_report.assert_called_once_with()
    # generate_report records its own report node; delegate must not add one.
    fake_agent.recorder.record_report.assert_not_called()


def test_delegate_no_pipeline_report_falls_back_to_generate_report(mocker):
    """A cutoff with no pipeline report (e.g. aborted run) falls back to synthesis."""
    fake_agent = _make_fake_agent(
        mocker, stop_reason="aborted", report="fallback report"
    )
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    out = ToolsRegistry().delegate(task="do X", context="ctx")
    assert out == "fallback report"
    fake_agent.generate_report.assert_called_once_with()


def test_delegate_marks_subagent_needs_handback(mocker):
    """The delegate subagent is constructed owing a hand-back report."""
    fake_agent = _make_fake_agent(mocker)
    spy = mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    ToolsRegistry().delegate(task="do X", context="ctx")
    assert spy.call_args.kwargs["needs_handback"] is True


def test_delegate_excludes_delegate_tool_from_subagent(mocker):
    """The subagent must not receive the delegate tool to prevent recursion."""
    fake_agent = _make_fake_agent(mocker)
    spy = mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    tools = [
        {"function": {"name": "bash"}},
        {"function": {"name": "delegate"}},
    ]
    reg = ToolsRegistry()
    reg._tools = tools
    ToolsRegistry.delegate(reg, task="t", context="c")
    _, kwargs = spy.call_args
    passed_tools = kwargs["tools"]
    names = [t["function"]["name"] for t in passed_tools]
    assert "delegate" not in names
    assert "bash" in names


def test_delegate_forwards_parent_toolset_via_executor(mocker, bare_llm):
    """Regression: a subagent spawned through the executor's registry receives
    the parent toolset minus ``delegate`` — never an empty list.

    Guards the zero-tools bug end-to-end: ToolExecutor must forward its toolset
    so ``delegate`` has a populated ``_tools`` to pass down.
    """
    from my_coding_agent.engine.tool_execution import ToolExecutor

    fake_agent = _make_fake_agent(mocker)
    spy = mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    tools = [
        {"function": {"name": "bash"}},
        {"function": {"name": "read_file"}},
        {"function": {"name": "delegate"}},
    ]
    executor = ToolExecutor({"tool_calls": []}, bare_llm, tools=tools)
    executor.registry.delegate(task="t", context="c")
    _, kwargs = spy.call_args
    names = [t["function"]["name"] for t in kwargs["tools"]]
    assert names == ["bash", "read_file"]

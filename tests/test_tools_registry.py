"""Tests for ToolsRegistry tool methods.

Filesystem is exercised via tmp_path (real but isolated and cheap). subprocess
and httpx are mocked so no real command runs and no real network call is made.
"""

import json
import subprocess

import httpx
import pytest

from my_coding_agent.engine.agent import DEFAULT_MAX_STEPS
from my_coding_agent.engine.schema import (
    REPORT_SOURCE_FALLBACK,
    REPORT_SOURCE_SUMMARIZER,
    REPORT_SOURCE_VERBATIM,
)
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


# --- read_tool_artifact: byte-range retrieval --------------------------------


def test_read_tool_artifact_range_returns_exact_slice():
    """A single-line artifact larger than the offload threshold is readable in
    exact, bounded pieces via start/length — no LLM call needed."""
    from my_coding_agent.engine.tool_registry.registry import RANGE_MAX_CHARS

    body = "x" * 100 + "y" * 100 + "z" * 100
    reg = ToolsRegistry(artifacts={"call_1": {"stdout": body, "stderr": ""}})
    out = reg.read_tool_artifact("call_1", start=100, length=100)
    assert out == f"[range 100-200 of {len(body)} bytes]\n" + "y" * 100
    assert RANGE_MAX_CHARS > 0  # sanity: budget constant is importable and positive


def test_read_tool_artifact_range_ignores_query():
    reg = ToolsRegistry(artifacts={"call_1": {"stdout": "0123456789", "stderr": ""}})
    out = reg.read_tool_artifact("call_1", query="ignored", start=0, length=5)
    assert out == "[range 0-5 of 10 bytes]\n01234"


def test_read_tool_artifact_range_defaults_length_to_budget():
    from my_coding_agent.engine.tool_registry.registry import RANGE_MAX_CHARS

    body = "a" * (RANGE_MAX_CHARS + 500)
    reg = ToolsRegistry(artifacts={"call_1": {"stdout": body, "stderr": ""}})
    out = reg.read_tool_artifact("call_1", start=0)
    header, _, slice_ = out.partition("\n")
    assert header == f"[range 0-{RANGE_MAX_CHARS} of {len(body)} bytes]"
    assert len(slice_) == RANGE_MAX_CHARS


def test_read_tool_artifact_range_length_is_capped_at_budget():
    from my_coding_agent.engine.tool_registry.registry import RANGE_MAX_CHARS

    body = "a" * (RANGE_MAX_CHARS + 500)
    reg = ToolsRegistry(artifacts={"call_1": {"stdout": body, "stderr": ""}})
    out = reg.read_tool_artifact("call_1", start=0, length=RANGE_MAX_CHARS + 500)
    header = out.splitlines()[0]
    assert header == f"[range 0-{RANGE_MAX_CHARS} of {len(body)} bytes]"


def test_read_tool_artifact_range_out_of_range_start_errors_with_total_size():
    reg = ToolsRegistry(artifacts={"call_1": {"stdout": "0123456789", "stderr": ""}})
    out = reg.read_tool_artifact("call_1", start=10)
    assert "Error" in out and "10 bytes" in out


def test_read_tool_artifact_range_missing_artifact_errors():
    reg = ToolsRegistry(artifacts={})
    out = reg.read_tool_artifact("nope", start=0)
    assert "no artifact found" in out


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


# --- extract-completeness-disclosure -----------------------------------------


def test_extract_chunk_detects_length_finish_reason(mocker):
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_OUTPUT_TOKEN_BUDGET,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [{"message": {"content": "cut mid"}, "finish_reason": "length"}],
            "usage": {"completion_tokens": EXTRACTION_OUTPUT_TOKEN_BUDGET},
        }
    )
    reg = ToolsRegistry(llm=fake_llm)
    extract, cut = reg._extract_chunk("chunk text", "q")
    assert extract == "cut mid"
    assert cut is True


def test_extract_chunk_detects_cap_without_finish_reason(mocker):
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_OUTPUT_TOKEN_BUDGET,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [{"message": {"content": "cut mid"}, "finish_reason": ""}],
            "usage": {"completion_tokens": EXTRACTION_OUTPUT_TOKEN_BUDGET},
        }
    )
    reg = ToolsRegistry(llm=fake_llm)
    _, cut = reg._extract_chunk("chunk text", "q")
    assert cut is True


def test_extract_chunk_trusts_explicit_stop(mocker):
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_OUTPUT_TOKEN_BUDGET,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [{"message": {"content": "clean"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": EXTRACTION_OUTPUT_TOKEN_BUDGET},
        }
    )
    reg = ToolsRegistry(llm=fake_llm)
    _, cut = reg._extract_chunk("chunk text", "q")
    assert cut is False


def test_extract_discloses_unscanned_remainder(mocker):
    """Output budget fills after the first of several chunks; the disclosure
    must name the scanned-vs-total extent and a recovery path."""
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_CHUNK_MAX_CHARS,
        EXTRACTION_OUTPUT_MAX_CHARS,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [
                {
                    "message": {"content": "x" * EXTRACTION_OUTPUT_MAX_CHARS},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 10},
        }
    )
    huge = "y" * (EXTRACTION_CHUNK_MAX_CHARS * 3)
    reg = ToolsRegistry(artifacts={"call_1": huge}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "detail near the end")
    assert "[Extract incomplete" in out
    assert f"of {len(huge)} stored characters" in out
    assert "follow-up query" in out


def test_extract_slice_disclosed_and_bounded(mocker):
    """A joined result exceeding the output budget is sliced and disclosed,
    with content + disclosure together staying within the budget."""
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_OUTPUT_MAX_CHARS,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [
                {
                    "message": {"content": "x" * (EXTRACTION_OUTPUT_MAX_CHARS + 1)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 10},
        }
    )
    reg = ToolsRegistry(artifacts={"call_1": "some stored text"}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "anything")
    assert len(out) <= EXTRACTION_OUTPUT_MAX_CHARS
    assert "[Extract incomplete" in out
    assert "output budget" in out


def test_extract_discloses_capped_chunk(mocker):
    """A capped chunk completion surfaces the cut disclosure in the return."""
    from my_coding_agent.engine.tool_registry.registry import (
        EXTRACTION_OUTPUT_TOKEN_BUDGET,
    )

    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [
                {
                    "message": {"content": "a short cut passage"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": EXTRACTION_OUTPUT_TOKEN_BUDGET},
        }
    )
    reg = ToolsRegistry(artifacts={"call_1": "some stored text"}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "anything")
    assert "a short cut passage" in out
    assert "[Extract incomplete" in out
    assert "token cap" in out


def test_complete_extract_unmarked(mocker):
    """All chunks scanned, clean finish, within budget — no false-positive marker."""
    fake_llm = mocker.Mock()
    fake_llm.chat_completion.return_value = mocker.Mock(
        json=lambda: {
            "choices": [
                {
                    "message": {"content": "the relevant passage"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 10},
        }
    )
    reg = ToolsRegistry(artifacts={"call_1": "some stored text"}, llm=fake_llm)
    out = reg.read_tool_artifact("call_1", "find the passage")
    assert out == "the relevant passage"
    assert "[Extract incomplete" not in out


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


def test_bash_stdin_delivers_data_byte_for_byte():
    hostile = 'it\'s "quoted" $(cmd) `backtick` <<HEREDOC\nline two\n'
    parsed = json.loads(ToolsRegistry().bash("cat", stdin=hostile))
    assert parsed["stdout"] == hostile.rstrip()
    assert parsed["ok"] is True


def test_bash_stdin_omitted_matches_prior_behavior(mocker):
    run = mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="echo hi", returncode=0, stdout="hi\n", stderr=""
        ),
    )
    ToolsRegistry().bash("echo hi")
    assert run.call_args.kwargs["input"] is None


def test_bash_stdin_empty_string_is_distinct_from_omitted(mocker):
    run = mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="cat", returncode=0, stdout="", stderr=""
        ),
    )
    ToolsRegistry().bash("cat", stdin="")
    assert run.call_args.kwargs["input"] == ""


def test_bash_stdin_ignored_by_command_completes_normally():
    parsed = json.loads(ToolsRegistry().bash("echo hi", stdin="unused input"))
    assert parsed["ok"] is True
    assert parsed["stdout"] == "hi"


def test_bash_stdin_with_timeout_returns_existing_timeout_envelope(mocker):
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 99", timeout=60),
    )
    parsed = json.loads(ToolsRegistry().bash("sleep 99", stdin="data"))
    assert parsed["ok"] is False
    assert parsed["exit_code"] == -1
    assert "timed out" in parsed["stderr"]


def test_bash_multiline_script_via_write_file_reads_stdin(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    script = tmp_path / "process.py"
    reg.write_file(
        "process.py",
        "import sys\ndata = sys.stdin.read()\nprint(data.strip().upper())\n",
    )
    parsed = json.loads(reg.bash(f"python3 {script}", stdin="hello world"))
    assert parsed["ok"] is True
    assert parsed["stdout"] == "HELLO WORLD"


# --- read_article ------------------------------------------------------------


def _mock_resp(mocker, text, content_type="text/html"):
    resp = mocker.Mock()
    resp.text = text
    resp.headers = {"content-type": content_type} if content_type is not None else {}
    resp.raise_for_status = mocker.Mock()
    return resp


def test_read_article_converts_html(mocker):
    resp = _mock_resp(mocker, "<h1>Title</h1><p>Body</p>", "text/html")
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert isinstance(out, tuple)
    assert "Title" in out[1]["stdout"]
    assert "Body" in out[1]["stdout"]
    assert out[1]["metadata"] == {
        "content_type": "text/html",
        "transform": "html-to-markdown",
    }


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
    resp = _mock_resp(mocker, "<p>" + ("word " * 20000) + "</p>", "text/html")
    # ~100_000 chars, well over ARTIFACT_THRESHOLD (8_000) but under the
    # fetch-side sanity cap.
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

    resp = _mock_resp(mocker, "<p>" + ("word " * 100_000) + "</p>", "text/html")
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert isinstance(out, tuple)
    assert "[...truncated" in out[1]["stdout"]
    assert len(out[1]["stdout"]) < ARTICLE_FETCH_MAX_CHARS + 200


# --- read_article: content-type fidelity (fetch-content-fidelity) ------------


def test_read_article_json_with_escapes_stays_parseable(mocker):
    """JSON body with backslash escapes and code blocks must round-trip through
    json.loads — the session fbef66a33c18 failure shape."""
    payload = {
        "note": "line1\\nline2\\ttabbed",
        "snippet": "```python\nprint('hi')\n```",
    }
    body = json.dumps(payload)
    resp = _mock_resp(mocker, body, "application/json")
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://api.example.com/data")
    assert isinstance(out, tuple)
    assert out[1]["stdout"] == body
    assert json.loads(out[1]["stdout"]) == payload
    assert out[1]["metadata"] == {
        "content_type": "application/json",
        "transform": "none",
    }


def test_read_article_plain_text_untouched(mocker):
    body = "*bold* _italic_ \\escaped # heading"
    resp = _mock_resp(mocker, body, "text/plain")
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com/notes.txt")
    assert out[1]["stdout"] == body
    assert out[1]["metadata"]["transform"] == "none"


def test_read_article_missing_content_type_is_verbatim(mocker):
    body = "raw content, no content-type header"
    resp = _mock_resp(mocker, body, content_type=None)
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com/unknown")
    assert out[1]["stdout"] == body
    assert out[1]["metadata"] == {"content_type": "unknown", "transform": "none"}


def test_read_article_binary_content_type_rejected(mocker):
    resp = _mock_resp(mocker, "\x89PNG...", "image/png")
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com/pic.png")
    assert isinstance(out, str)
    assert out.startswith("Error:")
    assert "image/png" in out


def test_read_article_verbatim_truncation_signaled_in_metadata_not_body(mocker):
    from my_coding_agent.engine.tool_registry.registry import ARTICLE_FETCH_MAX_CHARS

    body = json.dumps({"data": "x" * (ARTICLE_FETCH_MAX_CHARS + 1000)})
    resp = _mock_resp(mocker, body, "application/json")
    mocker.patch(
        "my_coding_agent.engine.tool_registry.registry.httpx.get", return_value=resp
    )
    out = ToolsRegistry.read_article("https://example.com/big.json")
    stdout = out[1]["stdout"]
    assert stdout == body[:ARTICLE_FETCH_MAX_CHARS]
    assert "truncated" not in stdout
    assert out[1]["metadata"]["truncated"] is True


# --- strip_task_restatements (redundancy guard) -------------------------------


def test_guard_drops_verbatim_task_copy():
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    task = "Collect latest CVEs from GitHub Advisories API as JSON and summarize."
    assert strip_task_restatements(task, task) == ""


def test_guard_drops_compressed_restatement_from_session_fbef66a33c18():
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    task = (
        "This task: collect latest CVEs from GitHub Advisories API as JSON "
        "and summarize them for the report."
    )
    restatement = (
        "Task: Collect latest CVEs from GitHub Advisories API as JSON and summarize."
    )
    assert strip_task_restatements(task, restatement) == ""


def test_guard_keeps_genuinely_additive_lines_unmodified():
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    task = "Read llm.py and explain how before_tool_call hooks work"
    facts = "Relevant files: agent.py, llm.py at /abs/path/repo"
    assert strip_task_restatements(task, facts) == facts


def test_guard_mixed_input_keeps_only_additive_lines():
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    task = "Read llm.py and explain how before_tool_call hooks work"
    facts = (
        "Read llm.py and explain how before_tool_call hooks work\n"
        "Working directory: /abs/path/repo"
    )
    result = strip_task_restatements(task, facts)
    assert result == "Working directory: /abs/path/repo"


def test_guard_all_restatement_input_yields_empty():
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    task = "Summarize the README file for this repo"
    facts = "Summarize the README\nfor this repo"
    assert strip_task_restatements(task, facts) == ""


def test_guard_keeps_short_additive_line_sharing_task_words():
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    task = "Check whether the service on port 8443 responds"
    facts = "use port 9000"
    assert strip_task_restatements(task, facts) == facts


def test_guard_uses_no_network_or_llm_calls(mocker):
    from my_coding_agent.engine.tool_registry.registry import (
        strip_task_restatements,
    )

    http_get = mocker.patch("httpx.get")
    strip_task_restatements("do the thing", "some fact\nanother fact")
    http_get.assert_not_called()


# --- delegate schema and opening-message construction -------------------------


def test_delegate_schema_known_facts_present_with_additive_description():
    from my_coding_agent.engine.tool_registry import function_to_json

    schema = function_to_json(ToolsRegistry.delegate)
    props = schema["function"]["parameters"]["properties"]
    assert "known_facts" in props
    description = props["known_facts"]["description"].lower()
    assert "not restate" in description or "do not restate" in description
    assert "file paths" in description


def test_delegate_schema_known_facts_not_required_task_is():
    from my_coding_agent.engine.tool_registry import function_to_json

    schema = function_to_json(ToolsRegistry.delegate)
    required = schema["function"]["parameters"]["required"]
    assert "task" in required
    assert "known_facts" not in required


def test_delegate_opening_message_task_only_when_known_facts_omitted(mocker):
    fake_agent = _make_fake_agent(mocker)
    captured = {}

    def _fake_agent_node(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return fake_agent

    mocker.patch("my_coding_agent.engine.agent.AgentNode", side_effect=_fake_agent_node)
    ToolsRegistry().delegate(task="do X")
    opening = captured["messages"][1]["content"]
    assert opening == "do X"


def test_delegate_opening_message_task_only_when_guard_empties_facts(mocker):
    fake_agent = _make_fake_agent(mocker)
    captured = {}

    def _fake_agent_node(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return fake_agent

    mocker.patch("my_coding_agent.engine.agent.AgentNode", side_effect=_fake_agent_node)
    ToolsRegistry().delegate(task="do X", known_facts="do X")
    opening = captured["messages"][1]["content"]
    assert opening == "do X"


def test_delegate_opening_message_presents_facts_distinctly_from_task(mocker):
    fake_agent = _make_fake_agent(mocker)
    captured = {}

    def _fake_agent_node(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return fake_agent

    mocker.patch("my_coding_agent.engine.agent.AgentNode", side_effect=_fake_agent_node)
    ToolsRegistry().delegate(task="do X", known_facts="file at /abs/path/repo")
    opening = captured["messages"][1]["content"]
    assert opening.count("do X") == 1
    assert "file at /abs/path/repo" in opening
    assert opening.index("do X") < opening.index("file at /abs/path/repo")


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
    out = ToolsRegistry().delegate(task="do X", known_facts="ctx")
    assert out == "final turn"
    fake_agent.execute.assert_called_once_with(max_steps=DEFAULT_MAX_STEPS)
    fake_agent.generate_report.assert_not_called()
    fake_agent.recorder.record_report.assert_called_once_with(
        "final turn", source=REPORT_SOURCE_VERBATIM
    )


def test_delegate_cutoff_returns_pipeline_report(mocker):
    """A cutoff hands back the pipeline-synthesized report, never the last turn."""
    fake_agent = _make_fake_agent(
        mocker, stop_reason="max_steps", handback_report="synthesized report"
    )
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    out = ToolsRegistry().delegate(task="do X", known_facts="ctx")
    assert out == "synthesized report"
    fake_agent.final_assistant_text.assert_not_called()
    fake_agent.generate_report.assert_not_called()
    fake_agent.recorder.record_report.assert_called_once_with(
        "synthesized report", source=REPORT_SOURCE_SUMMARIZER
    )


def test_delegate_empty_final_turn_falls_back_to_generate_report(mocker):
    """A clean stop with no usable final text falls back to synthesis."""
    fake_agent = _make_fake_agent(
        mocker, stop_reason="stop", final_text="", report="fallback report"
    )
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    out = ToolsRegistry().delegate(task="do X", known_facts="ctx")
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
    out = ToolsRegistry().delegate(task="do X", known_facts="ctx")
    assert out == "fallback report"
    fake_agent.generate_report.assert_called_once_with()


def _make_fake_agent_with_real_recorder(mocker, tmp_path, **kwargs):
    """Like ``_make_fake_agent``, but wired to a real ``Recorder`` writing into
    ``tmp_path`` so a test can read back the persisted ``report`` event and
    confirm its provenance from ``events.jsonl`` directly (no mock assertion)."""
    from my_coding_agent.engine.llm.schema import CALL_KIND_REPORT
    from my_coding_agent.observability.recorder import Recorder

    fake = _make_fake_agent(mocker, **kwargs)
    sdir = tmp_path / fake.session_id
    sdir.mkdir()
    fake.recorder = Recorder(session_id=fake.session_id, session_dir=sdir)

    def _fallback_report():
        fake.recorder.record_llm_call(
            kind=CALL_KIND_REPORT,
            call=1,
            latency_s=0.1,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            messages=[],
            context_window=8192,
            response_data={},
        )
        fake.recorder.record_report(
            kwargs.get("report", "report text"), source=REPORT_SOURCE_FALLBACK
        )
        return kwargs.get("report", "report text")

    fake.generate_report.side_effect = _fallback_report
    return fake, sdir / "events.jsonl"


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_delegate_clean_finish_end_to_end_zero_report_kind_rows(mocker, tmp_path):
    """Clean finish: verbatim source, no `report`-kind LLM call row exists."""
    fake_agent, events_path = _make_fake_agent_with_real_recorder(
        mocker, tmp_path, stop_reason="stop", final_text="final turn"
    )
    mocker.patch("my_coding_agent.engine.agent.AgentNode", return_value=fake_agent)

    ToolsRegistry().delegate(task="do X", known_facts="ctx")

    events = _read_events(events_path)
    report_events = [e for e in events if e["type"] == "report"]
    assert len(report_events) == 1
    assert report_events[0]["source"] == REPORT_SOURCE_VERBATIM
    assert not [e for e in events if e["type"] == "llm_call" and e["kind"] == "report"]


def test_delegate_cutoff_end_to_end_one_report_kind_row(mocker, tmp_path):
    """Cutoff with a pipeline hand-back: summarizer source, exactly one
    `report`-kind row (recorded in-pipeline before delegate() sees it)."""
    from my_coding_agent.engine.llm.schema import CALL_KIND_REPORT

    fake_agent, events_path = _make_fake_agent_with_real_recorder(
        mocker, tmp_path, stop_reason="max_steps", handback_report="synthesized report"
    )
    # Simulate ContextSummarizerNode already having recorded its report-kind
    # LLM call in-pipeline, before delegate() hands back the child's report.
    fake_agent.recorder.record_llm_call(
        kind=CALL_KIND_REPORT,
        call=1,
        latency_s=0.1,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        messages=[],
        context_window=8192,
        response_data={},
    )
    mocker.patch("my_coding_agent.engine.agent.AgentNode", return_value=fake_agent)

    ToolsRegistry().delegate(task="do X", known_facts="ctx")

    events = _read_events(events_path)
    report_events = [e for e in events if e["type"] == "report"]
    assert len(report_events) == 1
    assert report_events[0]["source"] == REPORT_SOURCE_SUMMARIZER
    report_kind_rows = [
        e for e in events if e["type"] == "llm_call" and e["kind"] == "report"
    ]
    assert len(report_kind_rows) == 1


def test_delegate_fallback_end_to_end_one_report_kind_row_and_resave(mocker, tmp_path):
    """Fallback path: fallback source, exactly one `report`-kind row, and the
    child record is re-saved so its persisted totals include that call."""
    fake_agent, events_path = _make_fake_agent_with_real_recorder(
        mocker, tmp_path, stop_reason="aborted", report="fallback report"
    )
    mocker.patch("my_coding_agent.engine.agent.AgentNode", return_value=fake_agent)

    out = ToolsRegistry().delegate(task="do X", known_facts="ctx")

    assert out == "fallback report"
    events = _read_events(events_path)
    report_events = [e for e in events if e["type"] == "report"]
    assert len(report_events) == 1
    assert report_events[0]["source"] == REPORT_SOURCE_FALLBACK
    report_kind_rows = [
        e for e in events if e["type"] == "llm_call" and e["kind"] == "report"
    ]
    assert len(report_kind_rows) == 1
    fake_agent._save_session_data.assert_called_once_with(DEFAULT_MAX_STEPS)


def test_delegate_resaves_session_data_after_generate_report(mocker):
    """D4: when generate_report() runs (out-of-pipeline), the child's session
    data is re-saved so its persisted total_usage includes the report call —
    execute() already saved once before the report was generated."""
    fake_agent = _make_fake_agent(
        mocker, stop_reason="aborted", report="fallback report"
    )
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    ToolsRegistry().delegate(task="do X", known_facts="ctx")
    fake_agent._save_session_data.assert_called_once_with(DEFAULT_MAX_STEPS)


def test_delegate_skips_resave_on_clean_finish(mocker):
    """No out-of-pipeline report call means no re-save is needed."""
    fake_agent = _make_fake_agent(mocker, stop_reason="stop", final_text="final turn")
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    ToolsRegistry().delegate(task="do X", known_facts="ctx")
    fake_agent._save_session_data.assert_not_called()


def test_delegate_hands_usage_summary_up_to_parent_agent_node(mocker):
    """D3: the completed child's usage summary reaches the parent via
    current_agent_node, without delegate() re-reading the child's files."""
    from my_coding_agent.observability.recorder import current_agent_node

    fake_agent = _make_fake_agent(mocker, stop_reason="stop", final_text="final turn")
    fake_agent._usage_summary.return_value = {"session_id": "abc123", "fake": True}
    mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    parent = mocker.Mock()
    token = current_agent_node.set(parent)
    try:
        ToolsRegistry().delegate(task="do X", known_facts="ctx")
    finally:
        current_agent_node.reset(token)
    parent.add_child_usage.assert_called_once_with(
        {"session_id": "abc123", "fake": True}
    )


def test_delegate_marks_subagent_needs_handback(mocker):
    """The delegate subagent is constructed owing a hand-back report."""
    fake_agent = _make_fake_agent(mocker)
    spy = mocker.patch(
        "my_coding_agent.engine.agent.AgentNode",
        return_value=fake_agent,
    )
    ToolsRegistry().delegate(task="do X", known_facts="ctx")
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
    ToolsRegistry.delegate(reg, task="t", known_facts="c")
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
    executor.registry.delegate(task="t", known_facts="c")
    _, kwargs = spy.call_args
    names = [t["function"]["name"] for t in kwargs["tools"]]
    assert names == ["bash", "read_file"]

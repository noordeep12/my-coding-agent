"""Tests for ToolsRegistry tool methods.

Filesystem is exercised via tmp_path (real but isolated and cheap). subprocess
and httpx are mocked so no real command runs and no real network call is made.
"""

import json
import subprocess

import httpx
import pytest

from my_coding_agent.tools import ARTIFACT_THRESHOLD, ToolsRegistry

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


def test_read_file_traversal_raises(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal detected"):
        reg.read_file("../../etc/passwd")


def test_write_file_absolute_outside_base_raises(tmp_path):
    reg = ToolsRegistry(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal detected"):
        reg.write_file("/tmp/escape.txt", "x")


# --- read_tool_artifact ------------------------------------------------------


def test_read_tool_artifact_found_dict():
    reg = ToolsRegistry(artifacts={"call_1": {"k": "v"}})
    assert json.loads(reg.read_tool_artifact("call_1")) == {"k": "v"}


def test_read_tool_artifact_found_str():
    reg = ToolsRegistry(artifacts={"call_1": "raw text"})
    assert reg.read_tool_artifact("call_1") == "raw text"


def test_read_tool_artifact_missing():
    reg = ToolsRegistry(artifacts={})
    out = reg.read_tool_artifact("nope")
    assert "no artifact found" in out


# --- bash --------------------------------------------------------------------


def test_bash_success_returns_json(mocker):
    mocker.patch(
        "my_coding_agent.tools.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args="echo hi", returncode=0, stdout="hi\n", stderr=""
        ),
    )
    out = ToolsRegistry().bash("echo hi")
    parsed = json.loads(out)
    assert parsed == {"stdout": "hi", "stderr": "", "exit_code": 0, "ok": True}


def test_bash_nonzero_exit_sets_ok_false(mocker):
    mocker.patch(
        "my_coding_agent.tools.subprocess.run",
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
        "my_coding_agent.tools.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 99", timeout=60),
    )
    parsed = json.loads(ToolsRegistry().bash("sleep 99"))
    assert parsed["ok"] is False
    assert parsed["exit_code"] == -1
    assert "timed out" in parsed["stderr"]


def test_bash_large_output_returns_artifact_tuple(mocker):
    big = "x" * (ARTIFACT_THRESHOLD + 1)
    mocker.patch(
        "my_coding_agent.tools.subprocess.run",
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
        "my_coding_agent.tools.subprocess.run",
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
    mocker.patch("my_coding_agent.tools.httpx.get", return_value=resp)
    out = ToolsRegistry.read_article("https://example.com")
    assert "Title" in out
    assert "Body" in out


def test_read_article_http_error(mocker):
    err_resp = mocker.Mock(status_code=404)
    mocker.patch(
        "my_coding_agent.tools.httpx.get",
        side_effect=httpx.HTTPStatusError("nf", request=mocker.Mock(), response=err_resp),
    )
    out = ToolsRegistry.read_article("https://example.com/missing")
    assert "HTTP 404" in out


def test_read_article_generic_error(mocker):
    mocker.patch(
        "my_coding_agent.tools.httpx.get", side_effect=httpx.ConnectError("down")
    )
    out = ToolsRegistry.read_article("https://example.com")
    assert out.startswith("Error fetching")


def test_read_article_truncates_long_content(mocker):
    resp = mocker.Mock()
    resp.text = "<p>" + ("word " * 20000) + "</p>"
    resp.raise_for_status = mocker.Mock()
    mocker.patch("my_coding_agent.tools.httpx.get", return_value=resp)
    out = ToolsRegistry.read_article("https://example.com")
    assert "truncated" in out

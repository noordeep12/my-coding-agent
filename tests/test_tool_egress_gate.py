"""Network egress filter at the tool-dispatch choke point (issue #126).

Locks in: a ``fetch_web`` call to a blocklisted host never reaches ``httpx``,
the model receives a structured ``ok:false`` envelope naming the host and the
matched list, a denial is recorded as a passive event, an allowed destination
is unaffected, and a disabled filter is byte-identical to today with no
egress events at all.
"""

from __future__ import annotations

import json

import httpx
import pytest

from my_coding_agent.engine.egress import schema as egress_schema
from my_coding_agent.engine.tool_registry import ToolRegistry as ToolsRegistry


@pytest.fixture(autouse=True)
def _no_real_http(monkeypatch):
    """Fail loudly if a blocked fetch ever reaches ``httpx.get``."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            f"httpx.get must never be called for a blocked destination: {args!r}"
        )

    monkeypatch.setattr(httpx, "get", _boom)


@pytest.fixture
def blocklist_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "blocklist.txt"
    cache_path.write_text("evil.example\n")
    monkeypatch.setenv(egress_schema.CACHE_PATH_ENV_VAR, str(cache_path))
    monkeypatch.delenv(egress_schema.DISABLE_ENV_VAR, raising=False)
    return cache_path


class TestGateBlocksExecution:
    def test_blocked_destination_makes_no_http_call(
        self, bare_executor, blocklist_cache
    ):
        raw, failure = bare_executor.invoke_tool(
            "c0", "fetch_web", {"url": "https://evil.example/page"}
        )
        assert raw is None
        assert failure is not None
        assert failure["reason"] == "egress_blocked"
        assert failure["block"].host == "evil.example"

    def test_unknown_destination_is_unaffected(
        self, bare_executor, blocklist_cache, monkeypatch
    ):
        monkeypatch.setattr(
            ToolsRegistry, "fetch_web", staticmethod(lambda **kw: "hello")
        )
        raw, failure = bare_executor.invoke_tool(
            "c0", "fetch_web", {"url": "https://good.example/page"}
        )
        assert failure is None
        assert raw == "hello"


class TestEgressEnvelope:
    def test_envelope_is_ok_false_with_structured_metadata(
        self, bare_executor, blocklist_cache
    ):
        _, failure = bare_executor.invoke_tool(
            "c0", "fetch_web", {"url": "https://evil.example/page"}
        )
        content, status, record = bare_executor.after_tool_call(
            "c0", "fetch_web", {"url": "https://evil.example/page"}, None, failure
        )
        env = json.loads(content)
        assert env["ok"] is False
        assert status == "error"
        assert record["ok"] is False
        assert "evil.example" in env["error"]
        assert "Blocked" in env["error"]
        egress_meta = env["metadata"]["egress"]
        assert egress_meta["host"] == "evil.example"
        assert egress_meta["matched_list"] == "hagezi"
        assert egress_meta["reason"]
        assert env["metadata"]["reason"] == "egress_blocked"

    def test_egress_emits_recorder_event_and_warning_log(
        self, bare_executor, blocklist_cache
    ):
        calls = []
        bare_executor.llm._recorder = type(
            "R",
            (),
            {
                "record_egress": lambda self, **kw: calls.append(kw),
                "after_tool": lambda self, *a, **kw: a[2] if a else None,
            },
        )()
        bare_executor.step_num = 5
        _, failure = bare_executor.invoke_tool(
            "c0", "fetch_web", {"url": "https://evil.example/page"}
        )
        bare_executor.after_tool_call(
            "c0", "fetch_web", {"url": "https://evil.example/page"}, None, failure
        )
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "fetch_web"
        assert calls[0]["host"] == "evil.example"
        assert calls[0]["matched_list"] == "hagezi"
        assert calls[0]["step"] == 5


class TestDisabledFilterIsByteIdentical:
    def test_disabled_filter_reaches_the_registry_unblocked(
        self, bare_executor, blocklist_cache, monkeypatch
    ):
        monkeypatch.setenv(egress_schema.DISABLE_ENV_VAR, "1")
        monkeypatch.setattr(
            ToolsRegistry, "fetch_web", staticmethod(lambda **kw: "hello")
        )
        raw, failure = bare_executor.invoke_tool(
            "c0", "fetch_web", {"url": "https://evil.example/page"}
        )
        assert failure is None
        assert raw == "hello"

    def test_disabled_filter_emits_no_egress_event(
        self, bare_executor, blocklist_cache, monkeypatch
    ):
        monkeypatch.setenv(egress_schema.DISABLE_ENV_VAR, "1")
        calls = []
        bare_executor.llm._recorder = type(
            "R",
            (),
            {
                "record_egress": lambda self, **kw: calls.append(kw),
                "after_tool": lambda self, *a, **kw: a[2] if a else None,
            },
        )()
        monkeypatch.setattr(
            ToolsRegistry, "fetch_web", staticmethod(lambda **kw: "hello")
        )
        raw, failure = bare_executor.invoke_tool(
            "c0", "fetch_web", {"url": "https://evil.example/page"}
        )
        bare_executor.after_tool_call(
            "c0", "fetch_web", {"url": "https://evil.example/page"}, raw, failure
        )
        assert calls == []

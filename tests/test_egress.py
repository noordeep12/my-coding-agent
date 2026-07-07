"""Network egress filter module (issue #126): blocklist load/cache/match.

Locks in: a known-bad host is blocked, an unknown host is allowed, matching
is parent-domain aware, an offline refresh falls back to the last-good cache,
an absent cache with a failed refresh fails open (never hard-blocks a run),
and a malformed list still parses to whatever valid lines it contains.
"""

from __future__ import annotations

import httpx
import pytest

from my_coding_agent.engine.egress import (
    EgressBlock,
    _host_matches,
    _parse_blocklist,
    evaluate,
    is_blocked,
    load_blocklist,
)
from my_coding_agent.engine.egress.schema import EgressConfig


def _config(tmp_path, **overrides):
    defaults = {
        "enabled": True,
        "source": "hagezi",
        "cache_path": tmp_path / "blocklist.txt",
        "refresh_cadence_hours": 24.0,
    }
    defaults.update(overrides)
    return EgressConfig(**defaults)


class TestHostMatching:
    def test_exact_host_matches(self):
        assert _host_matches("evil.example", {"evil.example"})

    def test_parent_domain_matches_subdomain(self):
        assert _host_matches("a.b.evil.example", {"evil.example"})

    def test_unrelated_host_does_not_match(self):
        assert not _host_matches("good.example", {"evil.example"})

    def test_lookalike_suffix_does_not_match(self):
        # "notevil.example" must not match a "evil.example" blocklist entry.
        assert not _host_matches("notevil.example", {"evil.example"})


class TestParseBlocklist:
    def test_plain_domain_per_line(self):
        text = "evil.example\nbad.example\n"
        assert _parse_blocklist(text) == {"evil.example", "bad.example"}

    def test_hostfile_format(self):
        text = "0.0.0.0 evil.example\n127.0.0.1 bad.example\n"
        assert _parse_blocklist(text) == {"evil.example", "bad.example"}

    def test_ignores_comments_and_blank_lines(self):
        text = "# comment\n\nevil.example\n"
        assert _parse_blocklist(text) == {"evil.example"}

    def test_malformed_lines_do_not_raise_and_valid_lines_survive(self):
        text = "evil.example\n   \n### header ###\nbad.example\n"
        assert _parse_blocklist(text) == {"evil.example", "bad.example"}


class TestIsBlocked:
    def test_known_bad_host_is_blocked(self, tmp_path):
        config = _config(tmp_path)
        config.cache_path.write_text("evil.example\n")
        result = is_blocked("evil.example", config)
        assert isinstance(result, EgressBlock)
        assert result.host == "evil.example"
        assert result.matched_list == "hagezi"

    def test_unknown_host_is_allowed(self, tmp_path):
        config = _config(tmp_path)
        config.cache_path.write_text("evil.example\n")
        assert is_blocked("good.example", config) is None

    def test_disabled_filter_allows_everything(self, tmp_path):
        config = _config(tmp_path, enabled=False)
        config.cache_path.write_text("evil.example\n")
        assert is_blocked("evil.example", config) is None


class TestOfflineTolerance:
    def test_offline_refresh_uses_last_good_cache(self, tmp_path, monkeypatch):
        config = _config(tmp_path, refresh_cadence_hours=0.0)
        config.cache_path.write_text("evil.example\n")

        def _boom(*args, **kwargs):
            raise httpx.ConnectError("network unreachable")

        monkeypatch.setattr(httpx, "get", _boom)
        domains = load_blocklist(config)
        assert domains == {"evil.example"}

    def test_absent_cache_and_failed_refresh_fails_open(self, tmp_path, monkeypatch):
        config = _config(tmp_path, refresh_cadence_hours=0.0)

        def _boom(*args, **kwargs):
            raise httpx.ConnectError("network unreachable")

        monkeypatch.setattr(httpx, "get", _boom)
        domains = load_blocklist(config)
        assert domains == set()
        # A downstream is_blocked check must not hard-fail the run either.
        assert is_blocked("anything.example", config) is None

    def test_fresh_cache_is_used_without_a_refetch(self, tmp_path, monkeypatch):
        config = _config(tmp_path, refresh_cadence_hours=999.0)
        config.cache_path.write_text("evil.example\n")

        def _boom(*args, **kwargs):
            raise AssertionError("httpx.get must not be called for a fresh cache")

        monkeypatch.setattr(httpx, "get", _boom)
        assert load_blocklist(config) == {"evil.example"}


class TestEvaluate:
    def test_non_fetch_web_tool_is_never_evaluated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCA_EGRESS_CACHE_PATH", str(tmp_path / "blocklist.txt"))
        assert evaluate("bash", {"command": "curl https://evil.example"}) is None

    def test_fetch_web_blocked_host(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCA_EGRESS_CACHE_PATH", str(tmp_path / "blocklist.txt"))
        (tmp_path / "blocklist.txt").write_text("evil.example\n")
        block = evaluate("fetch_web", {"url": "https://evil.example/page"})
        assert isinstance(block, EgressBlock)
        assert block.host == "evil.example"

    def test_fetch_web_allowed_host(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCA_EGRESS_CACHE_PATH", str(tmp_path / "blocklist.txt"))
        (tmp_path / "blocklist.txt").write_text("evil.example\n")
        assert evaluate("fetch_web", {"url": "https://good.example/page"}) is None

    @pytest.mark.parametrize("args", [{}, {"url": ""}, {"url": "not-a-url"}])
    def test_missing_or_hostless_url_is_never_blocked(self, args):
        assert evaluate("fetch_web", args) is None

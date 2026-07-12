"""Tests for webui/admin.py — LLM connection settings resolution and persistence."""

from __future__ import annotations

import io

from my_coding_agent.webui.admin import (
    build_llm_client,
    handle_admin_api_route,
    masked_llm_settings,
    resolve_llm_settings,
    save_llm_settings,
)
from my_coding_agent.webui.store import Store


def test_unset_fields_fall_back_to_env_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_API_URL", "http://env-host:1234/v1")
    monkeypatch.setenv("OMLX_MODEL", "env-model")
    monkeypatch.setenv("OMLX_API_KEY", "env-key")  # pragma: allowlist secret
    store = Store(tmp_path / "webui.db")
    resolved = resolve_llm_settings(store)
    assert resolved["api_url"] == "http://env-host:1234/v1"
    assert resolved["model"] == "env-model"
    assert resolved["api_key"] == "env-key"  # pragma: allowlist secret
    store.close()


def test_saved_value_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_API_URL", "http://env-host:1234/v1")
    store = Store(tmp_path / "webui.db")
    save_llm_settings(store, {"api_url": "http://saved-host:9999/v1"})
    resolved = resolve_llm_settings(store)
    assert resolved["api_url"] == "http://saved-host:9999/v1"
    store.close()


def test_round_trip_persists_across_store_reopen(tmp_path):
    db_path = tmp_path / "webui.db"
    store1 = Store(db_path)
    save_llm_settings(store1, {"api_url": "http://saved:1/v1", "model": "m1"})
    store1.close()

    store2 = Store(db_path)
    resolved = resolve_llm_settings(store2)
    assert resolved["api_url"] == "http://saved:1/v1"
    assert resolved["model"] == "m1"
    store2.close()


def test_save_ignores_empty_values(tmp_path):
    store = Store(tmp_path / "webui.db")
    save_llm_settings(store, {"api_url": "http://saved:1/v1"})
    save_llm_settings(store, {"api_url": "", "model": "m1"})
    resolved = resolve_llm_settings(store)
    assert resolved["api_url"] == "http://saved:1/v1"
    assert resolved["model"] == "m1"
    store.close()


def test_masked_settings_hide_api_key(tmp_path):
    store = Store(tmp_path / "webui.db")
    save_llm_settings(store, {"api_key": "supersecret"})  # pragma: allowlist secret
    masked = masked_llm_settings(store)
    assert masked["api_key"] == "********"
    assert "supersecret" not in str(masked)
    store.close()


def test_masked_settings_empty_key_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    store = Store(tmp_path / "webui.db")
    masked = masked_llm_settings(store)
    # falls back to the engine/llm default, which is non-empty, so masked too.
    assert masked["api_key"] == "********"
    store.close()


def test_build_llm_client_uses_resolved_settings(tmp_path):
    store = Store(tmp_path / "webui.db")
    save_llm_settings(
        store,
        {
            "api_url": "http://saved-host:9999/v1",
            "model": "saved-model",
            "api_key": "k",
        },
    )
    client = build_llm_client(store)
    assert client.api_url == "http://saved-host:9999/v1"
    assert client.model == "saved-model"
    assert client.api_key == "k"
    store.close()


def test_build_llm_client_falls_back_without_saved_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_MODEL", "env-model")
    store = Store(tmp_path / "webui.db")
    client = build_llm_client(store)
    assert client.model == "env-model"
    store.close()


# ── handle_admin_api_route ───────────────────────────────────────────────────


class _FakeHandler:
    """Just enough of BaseHTTPRequestHandler for handle_admin_api_route."""

    def __init__(self, body: bytes = b"") -> None:
        self.headers = {"Content-Length": str(len(body))} if body else {}
        self.rfile = io.BytesIO(body)
        self.sent: tuple[object, int] | None = None

    def _send_json(self, data, status=200):
        self.sent = (data, status)


def test_admin_route_ignores_other_paths(tmp_path):
    store = Store(tmp_path / "webui.db")
    handler = _FakeHandler()
    assert handle_admin_api_route(handler, "/api/other", "GET", store) is False
    assert handler.sent is None
    store.close()


def test_admin_route_get_returns_masked_settings(tmp_path):
    store = Store(tmp_path / "webui.db")
    save_llm_settings(store, {"api_key": "hushhush"})  # pragma: allowlist secret
    handler = _FakeHandler()
    assert handle_admin_api_route(handler, "/api/admin/settings", "GET", store) is True
    data, status = handler.sent
    assert status == 200
    assert data["api_key"] == "********"
    store.close()


def test_admin_route_post_saves_payload(tmp_path):
    store = Store(tmp_path / "webui.db")
    handler = _FakeHandler(b'{"model": "posted-model"}')
    assert handle_admin_api_route(handler, "/api/admin/settings", "POST", store) is True
    assert handler.sent == ({"ok": True}, 200)
    assert resolve_llm_settings(store)["model"] == "posted-model"
    store.close()


def test_admin_route_post_empty_body_saves_nothing(tmp_path):
    store = Store(tmp_path / "webui.db")
    handler = _FakeHandler()
    assert handle_admin_api_route(handler, "/api/admin/settings", "POST", store) is True
    assert handler.sent == ({"ok": True}, 200)
    store.close()


def test_admin_route_post_invalid_json_400(tmp_path):
    store = Store(tmp_path / "webui.db")
    handler = _FakeHandler(b"not json")
    assert handle_admin_api_route(handler, "/api/admin/settings", "POST", store) is True
    data, status = handler.sent
    assert status == 400
    assert data == {"error": "invalid json"}
    store.close()


def test_admin_route_post_non_dict_payload_400(tmp_path):
    store = Store(tmp_path / "webui.db")
    handler = _FakeHandler(b"[1, 2]")
    assert handle_admin_api_route(handler, "/api/admin/settings", "POST", store) is True
    data, status = handler.sent
    assert status == 400
    assert data == {"error": "invalid payload"}
    store.close()


def test_admin_route_unsupported_method_unhandled(tmp_path):
    store = Store(tmp_path / "webui.db")
    handler = _FakeHandler()
    assert handle_admin_api_route(handler, "/api/admin/settings", "PUT", store) is False
    assert handler.sent is None
    store.close()

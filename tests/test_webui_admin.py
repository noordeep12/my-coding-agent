"""Tests for webui/admin.py — LLM connection settings resolution and persistence."""

from __future__ import annotations

from my_coding_agent.webui.admin import (
    build_llm_client,
    masked_llm_settings,
    resolve_llm_settings,
    save_llm_settings,
)
from my_coding_agent.webui.store import Store


def test_unset_fields_fall_back_to_env_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_API_URL", "http://env-host:1234/v1")
    monkeypatch.setenv("OMLX_MODEL", "env-model")
    monkeypatch.setenv("OMLX_API_KEY", "env-key")
    store = Store(tmp_path / "webui.db")
    resolved = resolve_llm_settings(store)
    assert resolved["api_url"] == "http://env-host:1234/v1"
    assert resolved["model"] == "env-model"
    assert resolved["api_key"] == "env-key"
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
    save_llm_settings(store, {"api_key": "supersecret"})
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
        {"api_url": "http://saved-host:9999/v1", "model": "saved-model", "api_key": "k"},
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

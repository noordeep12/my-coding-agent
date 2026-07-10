"""Tests for webui/store.py — migrations, ui_state, and generic items CRUD."""

from __future__ import annotations

import sqlite3

from my_coding_agent.webui.store import Store, default_db_path


def test_fresh_db_migrates_to_current_version(tmp_path):
    store = Store(tmp_path / "webui.db")
    conn = sqlite3.connect(tmp_path / "webui.db")
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    store.close()
    assert version == 2
    assert {"ui_state", "items", "schema_version"} <= tables


def test_reopening_existing_db_is_a_noop(tmp_path):
    db_path = tmp_path / "webui.db"
    store1 = Store(db_path)
    store1.set_ui_state("route", {"tab": "traces"})
    store1.close()

    store2 = Store(db_path)
    assert store2.get_ui_state("route") == {"tab": "traces"}
    conn = sqlite3.connect(db_path)
    version = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    conn.close()
    store2.close()
    assert version == 1  # migration didn't re-insert a schema_version row


def test_crud_round_trips(tmp_path):
    store = Store(tmp_path / "webui.db")
    store.create_item("pipelines", "p1", {"name": "demo"})
    assert store.get_item("pipelines", "p1") == {"name": "demo"}

    store.update_item("pipelines", "p1", {"name": "demo2"})
    assert store.get_item("pipelines", "p1") == {"name": "demo2"}

    store.create_item("pipelines", "p2", {"name": "other"})
    assert len(store.list_items("pipelines")) == 2

    store.delete_item("pipelines", "p1")
    assert store.get_item("pipelines", "p1") is None
    assert len(store.list_items("pipelines")) == 1
    store.close()


def test_missing_ui_state_returns_none(tmp_path):
    store = Store(tmp_path / "webui.db")
    assert store.get_ui_state("nope") is None
    store.close()


def test_corrupt_ui_state_row_falls_back_cleanly(tmp_path):
    db_path = tmp_path / "webui.db"
    store = Store(db_path)
    store.close()
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO ui_state (key, value_json) VALUES ('route', 'not json')")
    conn.commit()
    conn.close()

    store2 = Store(db_path)
    assert store2.get_ui_state("route") is None
    store2.close()


def test_default_db_path(tmp_path):
    assert default_db_path(tmp_path) == tmp_path / "webui" / "webui.db"

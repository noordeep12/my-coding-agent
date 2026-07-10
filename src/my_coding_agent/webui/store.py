"""Durable local persistence for the webui shell.

A single SQLite file (stdlib `sqlite3`, no new dependency) holds a
`schema_version` table plus a forward-only migration runner, a generic
`items` CRUD table that feature tabs share by passing their own logical
`table_name`, and a `ui_state` key/value table backing restore-where-you-were.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Each entry is the SQL for migrating from version (index) to (index + 1).
_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE ui_state (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE items (
        table_name TEXT NOT NULL,
        id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (table_name, id)
    )
    """,
)


def default_db_path(base_dir: Path) -> Path:
    """Return the default store location under *base_dir* (`.my_coding_agent`)."""
    return base_dir / "webui" / "webui.db"


class Store:
    """SQLite-backed persistence for the webui shell and its feature tabs."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    # ── migrations ──────────────────────────────────────────────────────

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO schema_version (version) VALUES (0)")
                current = 0
            else:
                current = row["version"]
            for version, statements in enumerate(_MIGRATIONS, start=1):
                if version <= current:
                    continue
                self._conn.executescript(statements)
                self._conn.execute("UPDATE schema_version SET version = ?", (version,))

    # ── ui_state (restore-where-you-were) ──────────────────────────────

    def get_ui_state(self, key: str) -> dict[str, Any] | None:
        """Return the stored value for *key*, or `None` on a missing/corrupt row."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM ui_state WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            return dict(json.loads(row["value_json"]))
        except (ValueError, TypeError):
            return None

    def set_ui_state(self, key: str, value: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO ui_state (key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, json.dumps(value)),
            )

    # ── generic items CRUD (shared by feature tabs) ────────────────────

    def create_item(
        self, table_name: str, item_id: str, payload: dict[str, Any]
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO items "
                "(table_name, id, payload_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (table_name, item_id, json.dumps(payload), now, now),
            )

    def get_item(self, table_name: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM items WHERE table_name = ? AND id = ?",
                (table_name, item_id),
            ).fetchone()
        return json.loads(row["payload_json"]) if row is not None else None

    def update_item(
        self, table_name: str, item_id: str, payload: dict[str, Any]
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE items SET payload_json = ?, updated_at = ? "
                "WHERE table_name = ? AND id = ?",
                (json.dumps(payload), now, table_name, item_id),
            )

    def delete_item(self, table_name: str, item_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM items WHERE table_name = ? AND id = ?",
                (table_name, item_id),
            )

    def list_items(self, table_name: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload_json FROM items "
                "WHERE table_name = ? ORDER BY created_at",
                (table_name,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

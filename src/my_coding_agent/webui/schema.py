"""Typed row shapes for the webui persistence store (Schema Convention).

These dataclasses describe the logical rows the store CRUD layer returns;
they are not ORM models — `store.py` reads/writes plain dicts and this
module documents their shape for callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Current schema version. Bump and add a migration in `store.py` when the
#: physical table layout changes; wave-2 tabs adding their own logical
#: `items` tables (via `table_name`) do not need a schema bump.
SCHEMA_VERSION = 2


@dataclass
class UiStateRow:
    """One key/value row in `ui_state`, e.g. the last-visited route."""

    key: str
    value: dict[str, Any]


@dataclass
class ItemRow:
    """One row in the generic `items` CRUD table.

    `table_name` is a feature tab's logical table (e.g. "pipelines",
    "eval_configs"); `id` is caller-assigned.
    """

    table_name: str
    id: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str

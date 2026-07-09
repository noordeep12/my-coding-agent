"""Load eval cases from plain JSON files on disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .schema import EvalCase

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("id", "task", "scorer", "expected")


def load_case_set(case_dir: Path) -> list[EvalCase]:
    """Load every case in ``case_dir`` (one `*.json` file per case).

    A malformed case file (invalid JSON, missing a required key, or wrong
    field type) is skipped with a warning rather than failing the whole set.
    A duplicate case id within the set is likewise skipped with a warning,
    keeping the first occurrence.

    Args:
        case_dir: Directory containing one case per `*.json` file.

    Returns:
        Loaded cases, in file-name sort order. Empty if none load.
    """
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for path in sorted(case_dir.glob("*.json")):
        case = _load_case_file(path)
        if case is None:
            continue
        if case.id in seen_ids:
            logger.warning("skipping case %s: duplicate id in %s", path, case_dir)
            continue
        seen_ids.add(case.id)
        cases.append(case)
    return cases


def _load_case_file(path: Path) -> EvalCase | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("skipping malformed case %s: %s", path, exc)
        return None

    if not isinstance(raw, dict):
        logger.warning("skipping malformed case %s: not a JSON object", path)
        return None

    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        logger.warning("skipping malformed case %s: missing %s", path, missing)
        return None

    try:
        return EvalCase(
            id=str(raw["id"]),
            task=str(raw["task"]),
            scorer=str(raw["scorer"]),
            expected=dict(raw["expected"]),
            dataset=raw.get("dataset"),
            tags=tuple(raw.get("tags", ())),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("skipping malformed case %s: %s", path, exc)
        return None

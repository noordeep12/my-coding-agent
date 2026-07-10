"""Load eval cases from plain JSON files on disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .schema import EvalCase

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("id", "task", "scorer", "expected")


class CaseNotFoundError(Exception):
    """Raised when a case id has no file under the cases directory."""


def _case_path(case_id: str, case_dir: Path) -> Path:
    return case_dir / f"{case_id}.json"


def save_case(case: EvalCase, case_dir: Path) -> Path:
    """Create or overwrite the case file for ``case`` under ``case_dir``.

    Args:
        case: The case to write (id determines the file name).
        case_dir: Directory holding one `*.json` file per case.

    Returns:
        The path written.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    path = _case_path(case.id, case_dir)
    path.write_text(
        json.dumps(
            {
                "id": case.id,
                "task": case.task,
                "scorer": case.scorer,
                "expected": case.expected,
                "dataset": case.dataset,
                "tags": list(case.tags),
            },
            indent=2,
        )
        + "\n"
    )
    return path


def delete_case(case_id: str, case_dir: Path) -> None:
    """Delete the case file for ``case_id`` under ``case_dir``.

    Raises:
        CaseNotFoundError: If no case file exists for ``case_id``.
    """
    path = _case_path(case_id, case_dir)
    if not path.exists():
        raise CaseNotFoundError(f"No case '{case_id}' found under {case_dir}")
    path.unlink()


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

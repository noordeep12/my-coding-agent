"""Named, versioned collections of eval cases.

A dataset is a named, ordered collection of case ids with a version that
bumps every time membership changes (a case is added or retired). Prior
versions are never overwritten — each mutation appends a new version record
to the dataset's on-disk history, so an old version's membership stays
recoverable for comparisons that need to know exactly what a past run used.

Datasets reference cases by id only (not the case content). This keeps the
dataset model decoupled from the eval case file format defined in the eval
harness core change; a lister/runner that also has that format loaded can
resolve case ids to full ``EvalCase`` records.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATASETS_DIR = Path(".my_coding_agent/evals/datasets")
DEFAULT_CASES_DIR = Path(".my_coding_agent/evals/cases")

VERSIONS_FILE = "versions.jsonl"


class DatasetError(Exception):
    """Base exception for dataset operations."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset id has no on-disk history."""


class CaseNotInDatasetError(DatasetError):
    """Raised when retiring a case id that is not in the dataset's active set."""


@dataclass(frozen=True)
class Dataset:
    """A named, versioned, ordered collection of case ids.

    Args:
        id: Stable dataset identifier.
        version: Version number; bumps on every membership change.
        case_ids: Ordered, active case ids for this version.
    """

    id: str
    version: int
    case_ids: tuple[str, ...]


def _dataset_dir(dataset_id: str, base_dir: Path) -> Path:
    return base_dir / dataset_id


def _versions_path(dataset_id: str, base_dir: Path) -> Path:
    return _dataset_dir(dataset_id, base_dir) / VERSIONS_FILE


def _read_version_records(dataset_id: str, base_dir: Path) -> list[dict]:
    path = _versions_path(dataset_id, base_dir)
    if not path.exists():
        raise DatasetNotFoundError(
            f"No dataset '{dataset_id}' found under {base_dir}",
            hint="Create it first with create_dataset().",
        )
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _append_version_record(
    dataset_id: str,
    base_dir: Path,
    *,
    version: int,
    case_ids: tuple[str, ...],
    op: str,
    note: str = "",
) -> None:
    directory = _dataset_dir(dataset_id, base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "version": version,
        "case_ids": list(case_ids),
        "op": op,
        "note": note,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with _versions_path(dataset_id, base_dir).open("a") as f:
        f.write(json.dumps(record) + "\n")


def create_dataset(
    dataset_id: str,
    case_ids: Iterable[str],
    *,
    base_dir: Path = DEFAULT_DATASETS_DIR,
) -> Dataset:
    """Create a new dataset at version 1 with the given ordered case ids.

    Args:
        dataset_id: Stable identifier for the new dataset.
        case_ids: Ordered case ids forming the dataset's initial membership.
        base_dir: Directory holding one subdirectory per dataset.

    Returns:
        The new dataset at version 1.

    Raises:
        DatasetError: If a dataset with this id already exists.
    """
    if _versions_path(dataset_id, base_dir).exists():
        raise DatasetError(
            f"Dataset '{dataset_id}' already exists under {base_dir}",
            hint="Use add_case()/retire_case() to evolve an existing dataset.",
        )
    ordered = tuple(case_ids)
    _append_version_record(
        dataset_id, base_dir, version=1, case_ids=ordered, op="create"
    )
    return Dataset(id=dataset_id, version=1, case_ids=ordered)


def load_dataset(
    dataset_id: str,
    *,
    version: int | None = None,
    base_dir: Path = DEFAULT_DATASETS_DIR,
) -> Dataset:
    """Load a dataset's id, version, and ordered cases.

    Args:
        dataset_id: The dataset to load.
        version: A specific historical version to load; defaults to the
            latest (current) version.
        base_dir: Directory holding one subdirectory per dataset.

    Returns:
        The requested dataset version.

    Raises:
        DatasetNotFoundError: If the dataset id or the requested version
            does not exist.
    """
    records = _read_version_records(dataset_id, base_dir)
    if version is None:
        target = records[-1]
    else:
        matches = [r for r in records if r["version"] == version]
        if not matches:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' has no version {version}",
                hint=f"Known versions: {[r['version'] for r in records]}",
            )
        target = matches[-1]
    return Dataset(
        id=dataset_id, version=target["version"], case_ids=tuple(target["case_ids"])
    )


def add_case(
    dataset_id: str,
    case_id: str,
    *,
    base_dir: Path = DEFAULT_DATASETS_DIR,
    note: str = "",
) -> Dataset:
    """Add a case id to a dataset, producing a new version.

    Args:
        dataset_id: The dataset to mutate.
        case_id: The case id to add (appended to the ordered set).
        base_dir: Directory holding one subdirectory per dataset.
        note: Optional free-text note recorded with this version (e.g. why
            the case was added).

    Returns:
        The new dataset version with the case included.
    """
    current = load_dataset(dataset_id, base_dir=base_dir)
    new_case_ids = (*current.case_ids, case_id)
    new_version = current.version + 1
    _append_version_record(
        dataset_id,
        base_dir,
        version=new_version,
        case_ids=new_case_ids,
        op="add",
        note=note or f"add {case_id}",
    )
    return Dataset(id=dataset_id, version=new_version, case_ids=new_case_ids)


def retire_case(
    dataset_id: str,
    case_id: str,
    *,
    base_dir: Path = DEFAULT_DATASETS_DIR,
    note: str = "",
) -> Dataset:
    """Retire a case id from a dataset's active set, producing a new version.

    Membership history is preserved: the prior version (including the
    retired case) remains loadable via ``load_dataset(..., version=...)``.

    Args:
        dataset_id: The dataset to mutate.
        case_id: The case id to retire.
        base_dir: Directory holding one subdirectory per dataset.
        note: Optional free-text note recorded with this version.

    Returns:
        The new dataset version with the case removed.

    Raises:
        CaseNotInDatasetError: If the case id is not in the current version.
    """
    current = load_dataset(dataset_id, base_dir=base_dir)
    if case_id not in current.case_ids:
        raise CaseNotInDatasetError(
            f"Case '{case_id}' is not in dataset '{dataset_id}' "
            f"version {current.version}",
            hint=f"Active cases: {list(current.case_ids)}",
        )
    new_case_ids = tuple(c for c in current.case_ids if c != case_id)
    new_version = current.version + 1
    _append_version_record(
        dataset_id,
        base_dir,
        version=new_version,
        case_ids=new_case_ids,
        op="retire",
        note=note or f"retire {case_id}",
    )
    return Dataset(id=dataset_id, version=new_version, case_ids=new_case_ids)


def add_failure_case(
    dataset_id: str,
    *,
    case_id: str,
    task: str,
    scorer_ref: str,
    expected: dict,
    base_dir: Path = DEFAULT_DATASETS_DIR,
    cases_dir: Path = DEFAULT_CASES_DIR,
) -> Dataset:
    """Turn a recorded run failure into a new regression case and add it.

    Writes a case file (id, task, scorer ref, expected/threshold data — the
    same fields an ``EvalCase`` loads from disk) under ``cases_dir``, then
    adds its id to the dataset as a new version.

    Args:
        dataset_id: The dataset to add the regression case to.
        case_id: Stable id for the new case (must be unique under
            ``cases_dir``).
        task: The task prompt that produced the failure.
        scorer_ref: The scorer ref to judge the regression case.
        expected: The scorer's expected/threshold payload.
        base_dir: Directory holding one subdirectory per dataset.
        cases_dir: Directory eval cases load from.

    Returns:
        The new dataset version including the regression case.

    Raises:
        DatasetError: If a case file with this id already exists.
    """
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_path = cases_dir / f"{case_id}.json"
    if case_path.exists():
        raise DatasetError(
            f"Case file already exists at {case_path}",
            hint="Choose a unique case_id for the regression case.",
        )
    case_path.write_text(
        json.dumps(
            {
                "id": case_id,
                "task": task,
                "scorer_ref": scorer_ref,
                "expected": expected,
            },
            indent=2,
        )
        + "\n"
    )
    return add_case(
        dataset_id,
        case_id,
        base_dir=base_dir,
        note=f"regression case from run failure ({case_id})",
    )


def list_datasets(*, base_dir: Path = DEFAULT_DATASETS_DIR) -> list[Dataset]:
    """Enumerate available datasets and their current versions.

    Args:
        base_dir: Directory holding one subdirectory per dataset.

    Returns:
        One `Dataset` (at its latest version) per dataset found under
        `base_dir`, sorted by dataset id. Empty if `base_dir` doesn't exist.
    """
    if not base_dir.exists():
        return []
    datasets = []
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and (entry / VERSIONS_FILE).exists():
            datasets.append(load_dataset(entry.name, base_dir=base_dir))
    return datasets

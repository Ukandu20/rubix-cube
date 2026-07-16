"""Atomic JSON and CSV persistence for model artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a deterministic, human-readable JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_history_csv(
    path: Path | str,
    history: list[Mapping[str, Any]],
) -> None:
    """Atomically write update history, encoding nested values as JSON."""

    rows = list(history)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"history row {index} must be a mapping")

    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in rows:
        for fieldname in row:
            if not isinstance(fieldname, str):
                raise TypeError("history field names must be strings")
            if fieldname not in seen_fields:
                seen_fields.add(fieldname)
                fieldnames.append(fieldname)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {key: history_csv_value(row.get(key)) for key in fieldnames}
                    )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def history_csv_value(value: Any) -> Any:
    """Convert nested history values into stable compact JSON."""

    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value

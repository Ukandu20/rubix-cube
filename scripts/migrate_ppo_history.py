"""Move embedded PPO history from metrics.json files to adjacent CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from agents.ppo_agent import write_history_csv  # noqa: E402


def migrate_metrics_file(metrics_path: Path) -> bool:
    """Migrate one metrics file, returning whether it contained history."""

    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"{metrics_path}: metrics payload must be an object")
    if "history" not in payload:
        return False

    history = payload["history"]
    if not isinstance(history, list):
        raise TypeError(f"{metrics_path}: history must be a list")

    history_path = metrics_path.with_name("history.csv")
    write_history_csv(history_path, history)
    _verify_history_csv(history_path, history)

    migrated_payload = dict(payload)
    del migrated_payload["history"]
    _write_json_atomic(metrics_path, migrated_payload)
    return True


def migrate_tree(root: Path) -> tuple[int, int]:
    """Migrate all metrics files below root and return migrated/skipped counts."""

    migrated = 0
    skipped = 0
    for metrics_path in sorted(root.rglob("metrics.json")):
        if migrate_metrics_file(metrics_path):
            migrated += 1
        else:
            skipped += 1
    return migrated, skipped


def _verify_history_csv(
    history_path: Path,
    history: list[Mapping[str, Any]],
) -> None:
    expected_fields: list[str] = []
    seen_fields: set[str] = set()
    for row in history:
        for fieldname in row:
            if fieldname not in seen_fields:
                seen_fields.add(fieldname)
                expected_fields.append(fieldname)

    with history_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    actual_fields = reader.fieldnames or []
    if expected_fields != actual_fields:
        raise ValueError(f"{history_path}: CSV fields do not match history")
    if len(history) != len(rows):
        raise ValueError(f"{history_path}: CSV row count does not match history")

    for index, (source, output) in enumerate(zip(history, rows, strict=True)):
        if "evaluation" not in source:
            if output.get("evaluation", "") != "":
                raise ValueError(
                    f"{history_path}: unexpected evaluation at row {index}"
                )
            continue
        if json.loads(output["evaluation"]) != source["evaluation"]:
            raise ValueError(f"{history_path}: evaluation differs at row {index}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move PPO history arrays from metrics.json to history.csv."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "models" / "artifacts" / "ppo",
    )
    args = parser.parse_args()

    migrated, skipped = migrate_tree(args.root)
    print(f"Migrated {migrated} metrics files; skipped {skipped} without history.")


if __name__ == "__main__":
    main()

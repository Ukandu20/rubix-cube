"""Parquet behavior logging for cube-agent evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DEFAULT_BEHAVIOR_LOG_ROOT = Path("data/logs")


@dataclass(frozen=True)
class BehaviorLogConfig:
    """Output configuration for behavior logs."""

    model_version: str
    run_id: str
    logs_root: Path = DEFAULT_BEHAVIOR_LOG_ROOT


class BehaviorLogger:
    """Buffer episode and step rows and write them by scramble depth."""

    def __init__(self, config: BehaviorLogConfig) -> None:
        self.config = config
        self.episode_rows_by_depth: dict[int, list[dict[str, Any]]] = {}
        self.step_rows_by_depth: dict[int, list[dict[str, Any]]] = {}

    def log_episode(self, depth: int, row: Mapping[str, Any]) -> None:
        self.episode_rows_by_depth.setdefault(int(depth), []).append(dict(row))

    def log_step(self, depth: int, row: Mapping[str, Any]) -> None:
        self.step_rows_by_depth.setdefault(int(depth), []).append(dict(row))

    def flush(self) -> dict[str, list[Path]]:
        """Write buffered rows to parquet files and return written paths."""

        written_paths = {"episodes": [], "steps": []}
        for depth, rows in sorted(self.episode_rows_by_depth.items()):
            if not rows:
                continue
            path = self._depth_dir(depth) / "episodes.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path, index=False)
            written_paths["episodes"].append(path)

        for depth, rows in sorted(self.step_rows_by_depth.items()):
            if not rows:
                continue
            path = self._depth_dir(depth) / "steps.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path, index=False)
            written_paths["steps"].append(path)

        return written_paths

    def _depth_dir(self, depth: int) -> Path:
        return (
            Path(self.config.logs_root)
            / _safe_path_component(self.config.model_version)
            / _safe_path_component(self.config.run_id)
            / f"depth_{int(depth)}"
        )


def timestamp_run_id(prefix: str = "eval") -> str:
    """Return a filesystem-friendly timestamped run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def utc_timestamp() -> str:
    """Return an ISO timestamp for log rows."""

    return datetime.now(timezone.utc).isoformat()


def _safe_path_component(value: str) -> str:
    cleaned = str(value).strip().replace("\\", "_").replace("/", "_")
    return cleaned or "unknown"

"""Mixed-depth curriculum management for Rubik's Cube training."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cube.encoding import validate_encoded_state

SOLVED_STATE_STRING = "YYYYYYYYYOOOOOOOOOGGGGGGGGGWWWWWWWWWRRRRRRRRRBBBBBBBBB"


DEFAULT_CURRICULUM_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "curriculum_config_depth_1_5.yaml"
)


@dataclass(frozen=True)
class AdvancementThreshold:
    """Evaluation targets required to complete one curriculum level."""

    success_rate: float
    max_average_moves: float
    max_timeout_rate: float


@dataclass(frozen=True)
class CurriculumConfig:
    """Validated curriculum settings loaded from YAML."""

    min_depth: int
    max_depth: int
    starting_depth: int
    sampling_strategy: str
    mixed_sampling_weights: Mapping[int, Mapping[int, float]]
    advancement_thresholds: Mapping[int, AdvancementThreshold]
    max_steps_multiplier: int = 2
    max_steps_offset: int = 1

    def __post_init__(self) -> None:
        if self.min_depth <= 0:
            raise ValueError("curriculum min_depth must be positive")
        if self.max_depth < self.min_depth:
            raise ValueError("curriculum max_depth cannot be less than min_depth")
        if not self.min_depth <= self.starting_depth <= self.max_depth:
            raise ValueError("curriculum starting_depth is outside the depth range")
        if self.sampling_strategy not in ("mixed", "strict"):
            raise ValueError("sampling_strategy must be 'mixed' or 'strict'")
        if self.max_steps_multiplier <= 0 or self.max_steps_offset < 0:
            raise ValueError("curriculum max-step values must be non-negative")

        expected_depths = set(range(self.min_depth, self.max_depth + 1))
        if set(self.advancement_thresholds) != expected_depths:
            raise ValueError("advancement_thresholds must cover every curriculum depth")
        if self.sampling_strategy == "mixed":
            if set(self.mixed_sampling_weights) != expected_depths:
                raise ValueError(
                    "mixed_sampling_weights must cover every curriculum depth"
                )
            for level, weights in self.mixed_sampling_weights.items():
                if not weights:
                    raise ValueError(f"sampling weights for depth {level} are empty")
                invalid_depths = set(weights) - set(
                    range(self.min_depth, int(level) + 1)
                )
                if invalid_depths:
                    raise ValueError(
                        f"sampling weights for depth {level} include unavailable "
                        f"depths: {sorted(invalid_depths)}"
                    )
                if any(float(weight) < 0.0 for weight in weights.values()):
                    raise ValueError("mixed sampling weights cannot be negative")
                if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
                    raise ValueError(
                        f"sampling weights for depth {level} must sum to 1"
                    )

        for depth, threshold in self.advancement_thresholds.items():
            if not 0.0 <= threshold.success_rate <= 1.0:
                raise ValueError(f"invalid success rate threshold for depth {depth}")
            if threshold.max_average_moves <= 0:
                raise ValueError(f"invalid average-moves threshold for depth {depth}")
            if not 0.0 <= threshold.max_timeout_rate <= 1.0:
                raise ValueError(f"invalid timeout threshold for depth {depth}")

    def max_episode_steps(self, depth: int) -> int:
        """Return the episode cap for a curriculum level."""

        if not self.min_depth <= int(depth) <= self.max_depth:
            raise ValueError("depth is outside the curriculum range")
        return self.max_steps_multiplier * int(depth) + self.max_steps_offset

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""

        return asdict(self)


def load_curriculum_config(
    path: Path | str = DEFAULT_CURRICULUM_CONFIG_PATH,
) -> CurriculumConfig:
    """Load and validate curriculum settings from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    values = payload.get("curriculum", payload)
    if not isinstance(values, Mapping):
        raise ValueError("curriculum configuration must be a mapping")

    raw_weights = values.get("mixed_sampling_weights", {})
    weights = {
        int(level): {
            int(depth): float(weight) for depth, weight in level_weights.items()
        }
        for level, level_weights in raw_weights.items()
    }
    raw_thresholds = values.get("advancement_thresholds", {})
    thresholds = {
        int(depth): AdvancementThreshold(
            success_rate=float(target["success_rate"]),
            max_average_moves=float(target["max_average_moves"]),
            max_timeout_rate=float(target["max_timeout_rate"]),
        )
        for depth, target in raw_thresholds.items()
    }
    return CurriculumConfig(
        min_depth=int(values["min_depth"]),
        max_depth=int(values["max_depth"]),
        starting_depth=int(values.get("starting_depth", values["min_depth"])),
        sampling_strategy=str(values.get("sampling_strategy", "mixed")),
        mixed_sampling_weights=weights,
        advancement_thresholds=thresholds,
        max_steps_multiplier=int(values.get("max_steps_multiplier", 2)),
        max_steps_offset=int(values.get("max_steps_offset", 1)),
    )


class CurriculumManager:
    """Load depth datasets, sample states, and track curriculum progression."""

    def __init__(
        self,
        depth_files: Mapping[int, str | Path],
        config: CurriculumConfig,
        *,
        state_column: str = "state_encoded",
        exhaustive_state_threshold: int | None = 500,
        validate_dataset: bool = True,
        seed: int | None = None,
        warn_on_normalization: bool = True,
        state_data: Mapping[int, pd.DataFrame] | None = None,
    ) -> None:
        if exhaustive_state_threshold is not None and exhaustive_state_threshold <= 0:
            raise ValueError("exhaustive_state_threshold must be positive")

        self.config = config
        self.depth_files = {
            int(depth): Path(path) for depth, path in depth_files.items()
        }
        expected_depths = set(range(config.min_depth, config.max_depth + 1))
        missing = expected_depths - set(self.depth_files)
        if missing:
            raise ValueError(
                f"No precomputed states available for depth(s): {sorted(missing)}"
            )
        self.state_column = state_column
        self.exhaustive_state_threshold = exhaustive_state_threshold
        self.warn_on_normalization = warn_on_normalization
        self.current_depth = config.starting_depth
        self.cross_depth_duplicates_removed: dict[int, int] = {}
        self.solved_states_removed: dict[int, int] = {}
        if state_data is None:
            self.depth_data = self._load_depth_data(validate_dataset)
        else:
            supplied_depths = {int(depth) for depth in state_data}
            if supplied_depths != expected_depths:
                raise ValueError(
                    "state_data must cover every configured curriculum depth"
                )
            self.depth_data = {int(depth): frame for depth, frame in state_data.items()}
            if any(frame.empty for frame in self.depth_data.values()):
                raise ValueError("state_data cannot contain an empty depth frame")
        self._next_state_index_by_depth = {depth: 0 for depth in self.depth_data}
        self._rng = np.random.default_rng(seed)
        self.events: list[dict[str, Any]] = []

    def seed(self, seed: int) -> None:
        """Reset the manager's random generator without resetting cycle positions."""

        self._rng = np.random.default_rng(seed)

    def reset_depth_cursor(self, depth: int) -> None:
        """Restart exhaustive sampling for one configured depth."""

        selected_depth = int(depth)
        if selected_depth not in self.depth_data:
            raise ValueError(
                f"No precomputed states available for depth {selected_depth}."
            )
        self._next_state_index_by_depth[selected_depth] = 0

    def replace_depth_data(self, depth: int, frame: pd.DataFrame) -> None:
        """Replace a depth dataset and reset its sampling cursor safely."""

        selected_depth = int(depth)
        if selected_depth not in self.depth_data:
            raise ValueError(
                f"No precomputed states available for depth {selected_depth}."
            )
        if frame.empty:
            raise ValueError(f"depth {selected_depth} data cannot be empty")
        if self.state_column not in frame.columns:
            raise ValueError(
                f"depth {selected_depth} data is missing {self.state_column!r}"
            )
        self.depth_data[selected_depth] = frame.reset_index(drop=True).copy()
        self.reset_depth_cursor(selected_depth)

    def exclude_states(
        self,
        states_by_depth: Mapping[int, set[str] | list[str] | tuple[str, ...]],
    ) -> None:
        """Remove held-out states while keeping sampling state consistent."""

        for raw_depth, excluded_values in states_by_depth.items():
            depth = int(raw_depth)
            if depth not in self.depth_data:
                continue
            excluded = {str(value).strip() for value in excluded_values}
            if not excluded:
                continue
            frame = self.depth_data[depth]
            retained = frame.loc[
                ~frame[self.state_column].astype(str).str.strip().isin(excluded)
            ]
            if retained.empty:
                raise ValueError(
                    f"excluding held-out states leaves depth {depth} empty"
                )
            self.replace_depth_data(depth, retained)

    def sample_state(
        self,
        *,
        depth: int | None = None,
        state_index: int | None = None,
    ) -> tuple[int, pd.Series]:
        """Sample a row using the active curriculum and per-depth state strategy."""

        selected_depth = int(depth) if depth is not None else self._sample_depth()
        if selected_depth not in self.depth_data:
            raise ValueError(
                f"No precomputed states available for depth {selected_depth}."
            )
        frame = self.depth_data[selected_depth]

        if (
            state_index is None
            and self.state_selection_mode(selected_depth) == "exhaustive_cycle"
        ):
            state_index = self._next_state_index_by_depth[selected_depth]
            self._next_state_index_by_depth[selected_depth] = (state_index + 1) % len(
                frame
            )
        elif state_index is None:
            state_index = int(self._rng.integers(0, len(frame)))

        if state_index < 0 or state_index >= len(frame):
            raise ValueError(
                f"state_index must be between 0 and {len(frame) - 1} "
                f"for depth {selected_depth}"
            )
        return selected_depth, frame.iloc[state_index]

    def state_selection_mode(self, depth: int) -> str:
        """Return exhaustive cycling or random selection for one dataset."""

        if int(depth) not in self.depth_data:
            raise ValueError(f"No precomputed states available for depth {depth}.")
        return (
            "exhaustive_cycle"
            if (
                self.exhaustive_state_threshold is not None
                and len(self.depth_data[int(depth)]) < self.exhaustive_state_threshold
            )
            else "random"
        )

    def set_depth(self, depth: int) -> None:
        """Set the active curriculum level."""

        if not self.config.min_depth <= int(depth) <= self.config.max_depth:
            raise ValueError("Invalid curriculum depth")
        self.current_depth = int(depth)

    def get_current_depth(self) -> int:
        """Return the active curriculum level."""

        return self.current_depth

    def increase_depth(
        self,
        *,
        timestep: int | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> bool:
        """Advance one level and record the transition, if possible."""

        if self.current_depth >= self.config.max_depth:
            return False
        previous_depth = self.current_depth
        self.current_depth += 1
        self.events.append(
            {
                "from_depth": previous_depth,
                "to_depth": self.current_depth,
                "timestep": timestep,
                "metrics": dict(metrics or {}),
            }
        )
        return True

    def should_advance(self, metrics: Mapping[str, Any]) -> bool:
        """Return whether current-depth evaluation passes all targets."""

        if self.current_depth >= self.config.max_depth:
            return False
        average_moves = metrics.get(
            "average_solution_length",
            metrics.get("average_moves"),
        )
        if average_moves is None:
            return False
        success_rate = float(
            metrics.get("solve_rate", metrics.get("success_rate", 0.0))
        )
        timeout_rate = float(metrics.get("timeout_rate", 1.0))
        target = self.config.advancement_thresholds[self.current_depth]
        return (
            success_rate >= target.success_rate
            and float(average_moves) <= target.max_average_moves
            and timeout_rate <= target.max_timeout_rate
        )

    def current_weights(self) -> dict[int, float]:
        """Return the active depth probabilities."""

        if self.config.sampling_strategy == "strict":
            return {self.current_depth: 1.0}
        return {
            int(depth): float(weight)
            for depth, weight in self.config.mixed_sampling_weights[
                self.current_depth
            ].items()
        }

    def progress(self) -> dict[str, Any]:
        """Return serializable curriculum state."""

        return {
            "current_depth": self.current_depth,
            "max_depth": self.config.max_depth,
            "sampling_strategy": self.config.sampling_strategy,
            "current_weights": self.current_weights(),
            "events": list(self.events),
            "cross_depth_duplicates_removed": dict(self.cross_depth_duplicates_removed),
            "solved_states_removed": dict(self.solved_states_removed),
        }

    def _sample_depth(self) -> int:
        weights = self.current_weights()
        depths = np.asarray(list(weights), dtype=np.int64)
        probabilities = np.asarray(list(weights.values()), dtype=np.float64)
        return int(self._rng.choice(depths, p=probabilities))

    def _load_depth_data(self, validate_dataset: bool) -> dict[int, pd.DataFrame]:
        depth_data: dict[int, pd.DataFrame] = {}
        seen_states: set[str] = set()
        for depth in range(self.config.min_depth, self.config.max_depth + 1):
            path = self.depth_files[depth]
            frame = _read_state_file(path)
            if self.state_column not in frame.columns:
                raise ValueError(
                    f"{path} is missing required column {self.state_column!r}"
                )
            frame = frame.reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"Depth {depth} file contains no states: {path}")
            states = [str(value).strip() for value in frame[self.state_column]]
            if validate_dataset:
                if any(not validate_encoded_state(state) for state in states):
                    raise ValueError(
                        f"Depth {depth} file contains invalid encoded cube states: {path}"
                    )
                if bool(pd.Series(states).duplicated().any()):
                    raise ValueError(
                        f"Depth {depth} file contains duplicate states: {path}"
                    )
                cross_depth_mask = pd.Series(states).isin(seen_states).to_numpy()
                removed_count = int(cross_depth_mask.sum())
                if removed_count:
                    frame = frame.loc[~cross_depth_mask].reset_index(drop=True)
                    states = [str(value).strip() for value in frame[self.state_column]]
                    self.cross_depth_duplicates_removed[depth] = removed_count
                    if self.warn_on_normalization:
                        warnings.warn(
                            f"Removed {removed_count} state(s) from depth "
                            f"{depth} because they already occur at a shallower "
                            "curriculum depth.",
                            UserWarning,
                            stacklevel=2,
                        )
                    if frame.empty:
                        raise ValueError(
                            f"Depth {depth} contains no states after removing "
                            "states already available at shallower depths"
                        )
                solved_mask = pd.Series(states).eq(SOLVED_STATE_STRING).to_numpy()
                solved_count = int(solved_mask.sum())
                if solved_count:
                    frame = frame.loc[~solved_mask].reset_index(drop=True)
                    states = [str(value).strip() for value in frame[self.state_column]]
                    self.solved_states_removed[depth] = solved_count
                    if self.warn_on_normalization:
                        warnings.warn(
                            f"Removed {solved_count} solved state(s) from depth "
                            f"{depth}; solved states cannot start curriculum "
                            "episodes.",
                            UserWarning,
                            stacklevel=2,
                        )
                    if frame.empty:
                        raise ValueError(f"Depth {depth} contains no unsolved states")
            seen_states.update(states)
            depth_data[depth] = frame
        return depth_data


def _read_state_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"state file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported state file format: {path}")

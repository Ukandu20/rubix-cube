"""Gymnasium environment for dataset-driven Rubik's Cube solving."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

from cube.encoding import (
    STICKER_COUNT,
    decode_state,
    encode_state,
    validate_encoded_state,
)
from cube.environment import ACTION_TO_MOVE
from cube.episode import RewardConfig, advance_episode
from cube.state import CubeState

ENV_ID = "RubixCubeSolve-v0"
SOLVED_STATE_STRING = "YYYYYYYYYOOOOOOOOOGGGGGGGGGWWWWWWWWWRRRRRRRRRBBBBBBBBB"
DEFAULT_MAX_EPISODE_STEPS = 50
DEFAULT_EXHAUSTIVE_STATE_THRESHOLD = 10000
DEFAULT_TRAINING_DATA_DIR = Path("data/processed/training/parquet")
DEFAULT_STATE_FILES = {
    depth: DEFAULT_TRAINING_DATA_DIR / f"depth_{depth}.parquet" for depth in range(1, 6)
}
INVERSE_ACTION = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    6: 7,
    7: 6,
    8: 9,
    9: 8,
    10: 11,
    11: 10,
}
DEFAULT_REWARD_CONFIG = {
    "move_penalty": -0.01,
    "solve_bonus": 1.0,
    "inverse_move_penalty": -0.05,
    "timeout_penalty": -0.1,
}
OPTIONAL_INFO_COLUMNS = (
    "sample_id",
    "scramble_depth",
    "scramble_moves",
    "solution_moves",
    "first_solution_move",
)


class RubixCubeSolveEnv(gym.Env):
    """Dataset-driven Gymnasium environment for solving 3x3 cube states."""

    metadata = {"render_modes": ["text", "human"], "render_fps": 4}

    def __init__(
        self,
        state_files: Mapping[int, str | Path] | None = None,
        scramble_depth: int | None = 1,
        scramble_depth_range: tuple[int, int] | None = None,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        state_column: str = "state_encoded",
        reward_config: Mapping[str, float] | None = None,
        render_mode: str | None = None,
        validate_dataset: bool = True,
        exhaustive_state_threshold: int | None = None,
        curriculum_manager: Any | None = None,
        state_data: Mapping[int, pd.DataFrame] | None = None,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in (None, "text", "human"):
            raise ValueError("render_mode must be one of None, 'text', or 'human'")
        if exhaustive_state_threshold is not None and exhaustive_state_threshold <= 0:
            raise ValueError("exhaustive_state_threshold must be positive")

        configured_state_files = state_files
        if configured_state_files is None and curriculum_manager is not None:
            configured_state_files = curriculum_manager.depth_files
        if configured_state_files is None:
            configured_state_files = _default_state_files_for(
                scramble_depth,
                scramble_depth_range,
            )
        self.state_files = {
            int(depth): Path(path) for depth, path in configured_state_files.items()
        }
        if not self.state_files:
            raise ValueError("state_files cannot be empty")

        self.scramble_depth = scramble_depth
        self.scramble_depth_range = scramble_depth_range
        self.curriculum_manager = curriculum_manager
        self.max_episode_steps = max_episode_steps
        self.episode_max_steps = max_episode_steps
        self.state_column = state_column
        self.render_mode = render_mode
        self.exhaustive_state_threshold = exhaustive_state_threshold
        self.reward_config = {
            **DEFAULT_REWARD_CONFIG,
            **dict(reward_config or {}),
        }
        self.move_penalty = float(self.reward_config["move_penalty"])
        self.solve_bonus = float(self.reward_config["solve_bonus"])
        self.inverse_move_penalty = float(self.reward_config["inverse_move_penalty"])
        self.timeout_penalty = float(self.reward_config["timeout_penalty"])

        self.observation_space = spaces.Box(
            low=0,
            high=5,
            shape=(STICKER_COUNT,),
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(len(ACTION_TO_MOVE))
        self.solved_state = decode_state(SOLVED_STATE_STRING)

        if curriculum_manager is not None:
            self.states_by_depth = curriculum_manager.depth_data
        elif state_data is not None:
            self.states_by_depth = {
                int(depth): frame.reset_index(drop=True).copy()
                for depth, frame in state_data.items()
            }
        else:
            self.states_by_depth = self._load_state_files(validate_dataset)
        self._validate_depth_configuration()
        self._next_state_index_by_depth = {depth: 0 for depth in self.states_by_depth}

        self.cube = CubeState.solved()
        self.cube_state = self.solved_state.copy()
        self.current_step = 0
        self.current_depth = 0
        self.move_history: list[int] = []
        self.last_action: int | None = None
        self.last_move: str | None = None
        self.episode_return = 0.0
        self.episode_start_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode from one precomputed cube state."""

        super().reset(seed=seed)
        if self.curriculum_manager is not None and seed is not None:
            self.curriculum_manager.seed(seed)
        self.current_step = 0
        self.move_history = []
        self.last_action = None
        self.last_move = None
        self.episode_return = 0.0
        curriculum_depth = None
        if self.curriculum_manager is not None:
            curriculum_depth = self.curriculum_manager.get_current_depth()
            requested_depth = (
                int(options["scramble_depth"])
                if options and "scramble_depth" in options
                else None
            )
            state_index = (
                int(options["state_index"])
                if options and "state_index" in options
                else None
            )
            self.current_depth, row = self.curriculum_manager.sample_state(
                depth=requested_depth,
                state_index=state_index,
            )
            self.episode_max_steps = self.curriculum_manager.config.max_episode_steps(
                curriculum_depth
            )
        else:
            self.current_depth = self._select_depth(options)
            self.episode_max_steps = self.max_episode_steps
            state_index = None
            if options and "state_index" in options:
                state_index = int(options["state_index"])
            elif self.state_selection_mode(self.current_depth) == "exhaustive_cycle":
                state_index = self._next_state_index_by_depth[self.current_depth]
                self._next_state_index_by_depth[self.current_depth] = (
                    state_index + 1
                ) % len(self.states_by_depth[self.current_depth])
            row = self.sample_state_from_depth(
                self.current_depth,
                state_index=state_index,
            )
        encoded_state = str(row[self.state_column]).strip()
        self.cube = CubeState.from_flat_string(encoded_state, 3)
        self.cube_state = decode_state(encoded_state)
        self.episode_start_info = {
            **self._row_metadata(row),
            "sampled_depth": self.current_depth,
            "curriculum_depth": curriculum_depth,
        }

        info = {
            "state_source": "precomputed",
            "start_depth": self.current_depth,
            "sampled_depth": self.current_depth,
            "curriculum_depth": curriculum_depth,
            "encoded_state": encoded_state,
            "current_step": self.current_step,
            "max_episode_steps": self.episode_max_steps,
            "is_solved": self.is_solved(),
            **self.episode_start_info,
        }
        return self.get_observation(), info

    def exclude_states(
        self, states_by_depth: Mapping[int, set[str] | list[str]]
    ) -> None:
        """Remove held-out encoded states before training or evaluation resets."""

        for raw_depth, excluded_values in states_by_depth.items():
            depth = int(raw_depth)
            if depth not in self.states_by_depth:
                continue
            excluded = {str(value).strip() for value in excluded_values}
            if not excluded:
                continue
            frame = self.states_by_depth[depth]
            retained = frame.loc[
                ~frame[self.state_column].astype(str).str.strip().isin(excluded)
            ].reset_index(drop=True)
            if retained.empty:
                raise ValueError(
                    f"excluding held-out states leaves depth {depth} empty"
                )
            self.states_by_depth[depth] = retained
            if self.curriculum_manager is not None:
                self.curriculum_manager.replace_depth_data(depth, retained)
            self._next_state_index_by_depth[depth] = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one cube move and return the Gymnasium step tuple."""

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        action = int(action)
        transition = advance_episode(
            self.cube,
            action,
            action_to_move=ACTION_TO_MOVE,
            step_count=self.current_step,
            max_steps=self.episode_max_steps,
            previous_action=self.last_action,
            inverse_action=INVERSE_ACTION,
            reward_config=RewardConfig.from_mapping(self.reward_config),
        )
        self.cube = transition.cube
        self.cube_state = decode_state(self.cube.to_flat_string())
        self.current_step = transition.step_count
        self.move_history.append(action)

        terminated = transition.solved
        truncated = transition.timed_out

        self.last_action = action
        self.last_move = transition.move
        self.episode_return += transition.reward

        info = {
            "is_solved": transition.solved,
            "start_depth": self.current_depth,
            "sampled_depth": self.current_depth,
            "current_step": self.current_step,
            "max_episode_steps": self.episode_max_steps,
            "last_action": action,
            "last_move": transition.move,
            "immediate_inverse_move": transition.immediate_inverse,
            "move_history": self.move_history.copy(),
            "move_history_notation": [
                ACTION_TO_MOVE[action_id] for action_id in self.move_history
            ],
            "episode_return": self.episode_return,
            "terminated_reason": (
                "solved"
                if terminated
                else "max_steps_reached"
                if truncated
                else "running"
            ),
            **self.episode_start_info,
        }
        return self.get_observation(), transition.reward, terminated, truncated, info

    def sample_state_from_depth(
        self,
        depth: int,
        *,
        state_index: int | None = None,
    ) -> pd.Series:
        """Return one selected dataset row for a depth."""

        if depth not in self.states_by_depth:
            raise ValueError(f"No precomputed states available for depth {depth}.")

        frame = self.states_by_depth[depth]
        if frame.empty:
            raise ValueError(f"Depth {depth} file contains no states.")

        if state_index is not None:
            if state_index < 0 or state_index >= len(frame):
                raise ValueError(
                    f"state_index must be between 0 and {len(frame) - 1} "
                    f"for depth {depth}"
                )
            return frame.iloc[state_index]

        random_index = int(self.np_random.integers(0, len(frame)))
        return frame.iloc[random_index]

    def state_selection_mode(self, depth: int) -> str:
        """Return the configured state-selection strategy for a depth."""

        if self.curriculum_manager is not None:
            return self.curriculum_manager.state_selection_mode(depth)
        if depth not in self.states_by_depth:
            raise ValueError(f"No precomputed states available for depth {depth}.")
        available_states = len(self.states_by_depth[depth])
        return (
            "exhaustive_cycle"
            if (
                self.exhaustive_state_threshold is not None
                and available_states < self.exhaustive_state_threshold
            )
            else "random"
        )

    def calculate_reward(self, solved: bool, immediate_inverse: bool) -> float:
        """Calculate reward before any timeout penalty is added."""

        reward = self.move_penalty
        if immediate_inverse:
            reward += self.inverse_move_penalty
        if solved:
            reward += self.solve_bonus
        return reward

    def is_solved(self) -> bool:
        """Return whether the current cube matches the configured solved state."""

        return bool(np.array_equal(self.cube_state, self.solved_state))

    def get_observation(self) -> np.ndarray:
        """Return a defensive copy of the current observation."""

        return self.cube_state.copy()

    def set_curriculum_depth(self, depth: int) -> None:
        """Set the active curriculum level."""

        if self.curriculum_manager is None:
            raise RuntimeError("No curriculum manager is configured")
        self.curriculum_manager.set_depth(depth)

    def get_current_depth(self) -> int:
        """Return the active curriculum level or current sampled depth."""

        if self.curriculum_manager is not None:
            return self.curriculum_manager.get_current_depth()
        return self.current_depth

    def render(self) -> str | None:
        """Render the cube as text, printing it for human mode."""

        text = self._render_text()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def _select_depth(self, options: dict[str, Any] | None) -> int:
        if options and "scramble_depth" in options:
            depth = int(options["scramble_depth"])
        elif self.scramble_depth_range is not None:
            min_depth, max_depth = self.scramble_depth_range
            depth = int(self.np_random.integers(min_depth, max_depth + 1))
        else:
            if self.scramble_depth is None:
                raise ValueError(
                    "scramble_depth must be set when scramble_depth_range is None"
                )
            depth = int(self.scramble_depth)

        if depth not in self.states_by_depth:
            raise ValueError(f"No precomputed states available for depth {depth}.")
        return depth

    def _load_state_files(self, validate_dataset: bool) -> dict[int, pd.DataFrame]:
        states_by_depth: dict[int, pd.DataFrame] = {}
        seen_states: set[str] = set()

        for depth, path in sorted(self.state_files.items()):
            frame = _read_state_file(path)
            if self.state_column not in frame.columns:
                raise ValueError(
                    f"{path} is missing required column {self.state_column!r}"
                )
            if validate_dataset:
                self._validate_state_frame(depth, path, frame, seen_states)
            else:
                seen_states.update(
                    str(value).strip() for value in frame[self.state_column]
                )
            states_by_depth[depth] = frame.reset_index(drop=True)

        return states_by_depth

    def _validate_state_frame(
        self,
        depth: int,
        path: Path,
        frame: pd.DataFrame,
        seen_states: set[str],
    ) -> None:
        if frame.empty:
            raise ValueError(f"Depth {depth} file contains no states: {path}")

        states = [str(value).strip() for value in frame[self.state_column]]
        invalid_states = [
            state for state in states if not validate_encoded_state(state)
        ]
        if invalid_states:
            raise ValueError(
                f"Depth {depth} file contains invalid encoded cube states: {path}"
            )

        duplicates = pd.Series(states).duplicated()
        if bool(duplicates.any()):
            raise ValueError(f"Depth {depth} file contains duplicate states: {path}")

        cross_duplicates = set(states) & seen_states
        if cross_duplicates:
            raise ValueError(
                f"Depth {depth} file contains states already loaded from another depth"
            )

        if SOLVED_STATE_STRING in set(states):
            raise ValueError(f"Depth {depth} file contains the solved state: {path}")

        seen_states.update(states)

    def _validate_depth_configuration(self) -> None:
        if self.curriculum_manager is not None:
            return
        if self.scramble_depth_range is not None:
            min_depth, max_depth = self.scramble_depth_range
            if min_depth > max_depth:
                raise ValueError("scramble_depth_range minimum cannot exceed maximum")
            missing = [
                depth
                for depth in range(min_depth, max_depth + 1)
                if depth not in self.states_by_depth
            ]
            if missing:
                raise ValueError(
                    f"No precomputed states available for depth(s): {missing}"
                )
        elif self.scramble_depth is None:
            raise ValueError(
                "scramble_depth must be set when scramble_depth_range is None"
            )
        elif int(self.scramble_depth) not in self.states_by_depth:
            raise ValueError(
                f"No precomputed states available for depth {self.scramble_depth}."
            )

    def _row_metadata(self, row: pd.Series) -> dict[str, Any]:
        return {
            column: _clean_metadata_value(row.get(column))
            for column in OPTIONAL_INFO_COLUMNS
            if column in row.index
        }

    def _render_text(self) -> str:
        encoded_state = encode_state(self.cube_state)
        lines = [
            f"Step: {self.current_step} / {self.episode_max_steps}",
            f"Start depth: {self.current_depth}",
            f"Last move: {self.last_move}",
            f"Solved: {self.is_solved()}",
            "",
        ]
        for face_index in range(6):
            start = face_index * 9
            face = encoded_state[start : start + 9]
            lines.append(f"Face {face_index}:")
            for row_start in range(0, 9, 3):
                lines.append(" ".join(face[row_start : row_start + 3]))
            if face_index != 5:
                lines.append("")
        return "\n".join(lines)


def _read_state_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"state file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported state file format: {path}")


def _default_state_files_for(
    scramble_depth: int | None,
    scramble_depth_range: tuple[int, int] | None,
) -> dict[int, Path]:
    if scramble_depth_range is not None:
        min_depth, max_depth = scramble_depth_range
        return {
            depth: DEFAULT_STATE_FILES[depth]
            for depth in range(min_depth, max_depth + 1)
            if depth in DEFAULT_STATE_FILES
        }
    if scramble_depth is None:
        return dict(DEFAULT_STATE_FILES)
    depth = int(scramble_depth)
    if depth in DEFAULT_STATE_FILES:
        return {depth: DEFAULT_STATE_FILES[depth]}
    return dict(DEFAULT_STATE_FILES)


def _clean_metadata_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def register_gym_environment() -> None:
    """Register the environment ID with Gymnasium once."""

    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="cube.gym_environment:RubixCubeSolveEnv",
        )


register_gym_environment()

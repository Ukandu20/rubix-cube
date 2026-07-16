"""Reinforcement learning environment wrapper for cube states."""

from __future__ import annotations

import random

from cube.episode import RewardConfig, advance_episode
from cube.moves import apply_move
from cube.notation import generate_scramble
from cube.state import CubeState

ACTION_TO_MOVE: dict[int, str] = {
    0: "U",
    1: "U'",
    2: "R",
    3: "R'",
    4: "F",
    5: "F'",
    6: "D",
    7: "D'",
    8: "L",
    9: "L'",
    10: "B",
    11: "B'",
}
MOVE_TO_ACTION: dict[str, int] = {
    move: action for action, move in ACTION_TO_MOVE.items()
}
ACTION_SIZE = len(ACTION_TO_MOVE)
DEFAULT_MAX_STEPS = 30


class CubeEnvironment:
    """Small Gym-style environment for 3x3 cube episodes."""

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        rng: random.Random | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.max_steps = max_steps
        self.rng = rng
        self.cube = CubeState.solved()
        self.move_count = 0
        self.scramble_depth = 0
        self.scramble_sequence = ""
        self.move_history: list[str] = []

    def reset(self) -> str:
        """Reset the episode to a solved cube and return the observation."""

        self.cube = CubeState.solved()
        self.move_count = 0
        self.max_steps = DEFAULT_MAX_STEPS
        self.scramble_depth = 0
        self.scramble_sequence = ""
        self.move_history = []
        return self.get_state()

    def step(self, action: int) -> tuple[str, float, bool, dict]:
        """Apply one discrete action and return next_state, reward, done, info."""

        self._move_for_action(action)
        transition = advance_episode(
            self.cube,
            action,
            action_to_move=ACTION_TO_MOVE,
            step_count=self.move_count,
            max_steps=self.max_steps,
            reward_config=RewardConfig(),
        )
        self.cube = transition.cube
        self.move_count = transition.step_count
        self.move_history.append(transition.move)
        done = transition.solved or transition.timed_out
        info = {
            "move": transition.move,
            "move_count": self.move_count,
            "max_steps": self.max_steps,
            "is_solved": transition.solved,
            "scramble": self.scramble_sequence,
            "timeout": transition.timed_out,
        }
        return self.get_state(), transition.reward, done, info

    def scramble(self, depth: int) -> str:
        """Reset to solved, apply a generated scramble, and start a new episode."""

        if depth < 0:
            raise ValueError("scramble depth cannot be negative")

        self.cube = CubeState.solved()
        self.max_steps = _max_steps_for_depth(depth)
        self.scramble_depth = depth
        self.scramble_sequence = generate_scramble(
            length=depth,
            rng=self.rng,
        )
        if self.scramble_sequence:
            for token in self.scramble_sequence.split():
                self.cube = apply_move(self.cube, token)
        self.move_count = 0
        self.move_history = []
        return self.scramble_sequence

    def is_solved(self) -> bool:
        """Return whether the current cube is solved."""

        return self.cube.is_solved()

    def get_state(self) -> str:
        """Return the current flat-string observation."""

        return self.cube.to_flat_string()

    def render(self) -> str:
        """Return a text net of the current cube."""

        faces = self.cube.faces
        size = self.cube.size
        indent = " " * (size * 2)
        lines: list[str] = []

        for row in faces["U"]:
            lines.append(f"{indent}{_format_row(row)}")
        for row_index in range(size):
            lines.append(
                "   ".join(
                    _format_row(faces[face_name][row_index])
                    for face_name in ("L", "F", "R", "B")
                )
            )
        for row in faces["D"]:
            lines.append(f"{indent}{_format_row(row)}")

        return "\n".join(lines)

    def copy(self) -> CubeEnvironment:
        """Return an independent environment with the same episode state."""

        clone = CubeEnvironment(max_steps=self.max_steps, rng=self.rng)
        clone.cube = self.cube.copy()
        clone.move_count = self.move_count
        clone.scramble_depth = self.scramble_depth
        clone.scramble_sequence = self.scramble_sequence
        clone.move_history = list(self.move_history)
        return clone

    def clone(self) -> CubeEnvironment:
        """Alias for copy, useful for search algorithms."""

        return self.copy()

    @staticmethod
    def _move_for_action(action: int) -> str:
        if not isinstance(action, int):
            raise ValueError("action must be an integer")
        if action not in ACTION_TO_MOVE:
            raise ValueError(f"action must be between 0 and {ACTION_SIZE - 1}")
        return ACTION_TO_MOVE[action]


def _max_steps_for_depth(depth: int) -> int:
    if depth == 0:
        return DEFAULT_MAX_STEPS
    if 1 <= depth <= 3:
        return 10
    if 4 <= depth <= 6:
        return 20
    if 7 <= depth <= 10:
        return 30
    return max(30, depth * 3)


def _format_row(row: list[str]) -> str:
    return " ".join(row)

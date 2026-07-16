"""Shared episode-transition rules for cube environment adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cube.moves import apply_move
from cube.state import CubeState


@dataclass(frozen=True)
class RewardConfig:
    """Rewards applied to one environment transition."""

    move_penalty: float = -0.01
    solve_bonus: float = 1.01
    inverse_move_penalty: float = 0.0
    timeout_penalty: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> RewardConfig:
        """Build a reward configuration from optional named overrides."""

        if values is None:
            return cls()
        return cls(
            move_penalty=float(values.get("move_penalty", cls.move_penalty)),
            solve_bonus=float(values.get("solve_bonus", cls.solve_bonus)),
            inverse_move_penalty=float(
                values.get("inverse_move_penalty", cls.inverse_move_penalty)
            ),
            timeout_penalty=float(values.get("timeout_penalty", cls.timeout_penalty)),
        )


@dataclass(frozen=True)
class EpisodeTransition:
    """Framework-neutral result of applying one cube action."""

    cube: CubeState
    action: int
    move: str
    step_count: int
    solved: bool
    timed_out: bool
    immediate_inverse: bool
    reward: float


def advance_episode(
    cube: CubeState,
    action: int,
    *,
    action_to_move: Mapping[int, str],
    step_count: int,
    max_steps: int,
    previous_action: int | None = None,
    inverse_action: Mapping[int, int] | None = None,
    reward_config: RewardConfig | None = None,
) -> EpisodeTransition:
    """Apply an action and calculate shared solved, timeout, and reward rules."""

    if action not in action_to_move:
        raise ValueError(f"Invalid action: {action}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    config = reward_config or RewardConfig()
    move = action_to_move[action]
    next_cube = apply_move(cube, move)
    next_step = step_count + 1
    solved = next_cube.is_solved()
    timed_out = not solved and next_step >= max_steps
    immediate_inverse = bool(
        previous_action is not None
        and inverse_action is not None
        and action == inverse_action.get(previous_action)
    )

    reward = config.move_penalty
    if solved:
        reward += config.solve_bonus
    if immediate_inverse:
        reward += config.inverse_move_penalty
    if timed_out:
        reward += config.timeout_penalty

    return EpisodeTransition(
        cube=next_cube,
        action=action,
        move=move,
        step_count=next_step,
        solved=solved,
        timed_out=timed_out,
        immediate_inverse=immediate_inverse,
        reward=reward,
    )

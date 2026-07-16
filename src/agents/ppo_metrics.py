"""Evaluation and rolling-training metrics shared by PPO workflows."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from statistics import mean, median
from typing import Any

from cube.environment import ACTION_TO_MOVE


def evaluation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize episode-level evaluation records."""

    solved_rows = [row for row in rows if row["solved"]]
    solution_lengths = [
        int(row["solution_length"])
        for row in solved_rows
        if row["solution_length"] is not None
    ]
    extra_moves = [
        int(row["extra_moves"]) for row in solved_rows if row["extra_moves"] is not None
    ]
    total_steps = sum(int(row["steps"]) for row in rows)
    counts = action_counts(rows)
    return {
        "episodes": len(rows),
        "solved_count": len(solved_rows),
        "solve_rate": len(solved_rows) / len(rows) if rows else 0.0,
        "average_reward": mean(row["reward"] for row in rows) if rows else 0.0,
        "average_solution_length": mean(solution_lengths) if solution_lengths else None,
        "median_solution_length": median(solution_lengths)
        if solution_lengths
        else None,
        "timeout_rate": (
            sum(int(row["timeout"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "inverse_move_rate": (
            sum(int(row["inverse_moves"]) for row in rows) / total_steps
            if total_steps
            else 0.0
        ),
        "average_extra_moves": mean(extra_moves) if extra_moves else None,
        "action_counts": counts,
        "action_distribution": action_distribution(counts),
    }


def action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count selected move notation across evaluation records."""

    counts = {ACTION_TO_MOVE[action_id]: 0 for action_id in sorted(ACTION_TO_MOVE)}
    for row in rows:
        for move in row.get("moves_taken", []):
            if move in counts:
                counts[move] += 1
    return counts


def action_distribution(action_counts: Mapping[str, int]) -> dict[str, float]:
    """Normalize move counts into probabilities."""

    total = sum(int(count) for count in action_counts.values())
    if total == 0:
        return {move: 0.0 for move in action_counts}
    return {move: int(count) / total for move, count in action_counts.items()}


def episode_window_metrics(
    episodes: deque[dict[str, Any]],
) -> dict[str, float]:
    """Summarize the recent training-episode window."""

    if not episodes:
        return {
            "mean_episode_reward": 0.0,
            "mean_episode_length": 0.0,
            "train_solve_rate": 0.0,
            "timeout_rate": 0.0,
            "inverse_move_rate": 0.0,
        }
    total_steps = sum(int(row["length"]) for row in episodes)
    return {
        "mean_episode_reward": mean(float(row["reward"]) for row in episodes),
        "mean_episode_length": mean(int(row["length"]) for row in episodes),
        "train_solve_rate": mean(float(row["solved"]) for row in episodes),
        "timeout_rate": mean(float(row["timeout"]) for row in episodes),
        "inverse_move_rate": (
            sum(int(row["inverse_moves"]) for row in episodes) / total_steps
            if total_steps
            else 0.0
        ),
    }

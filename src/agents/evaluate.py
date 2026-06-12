"""Evaluation helpers for cube baseline agents."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

from agents.bfs_agent import BFSAgent
from agents.inverse_scramble_agent import InverseScrambleAgent
from agents.random_agent import RandomAgent
from cube.environment import CubeEnvironment
from cube.moves import apply_move


DEFAULT_DEPTHS = range(1, 6)
DEFAULT_CSV_PATH = Path("reports/baselines.csv")


def evaluate_random_agent(
    depths: Iterable[int] = DEFAULT_DEPTHS,
    episodes_per_depth: int = 100,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """Evaluate random actions using each environment episode limit."""

    _validate_episode_count(episodes_per_depth)
    random_source = rng if rng is not None else random.Random()
    agent = RandomAgent(rng=random_source)
    results: list[dict] = []

    for depth in depths:
        solved_moves: list[int] = []
        solved_count = 0
        start_time = perf_counter()
        for _ in range(episodes_per_depth):
            env = CubeEnvironment(rng=random_source)
            env.scramble(depth)
            done = env.is_solved()
            info = {"is_solved": done}
            while not done:
                _, _, done, info = env.step(agent.act(env))
            if info["is_solved"]:
                solved_count += 1
                solved_moves.append(env.move_count)

        elapsed = perf_counter() - start_time
        results.append(
            _base_metrics(
                agent="random",
                depth=depth,
                episodes=episodes_per_depth,
                solved_count=solved_count,
                elapsed_seconds=elapsed,
                solved_moves=solved_moves,
            )
        )

    return results


def evaluate_inverse_scramble_agent(
    depths: Iterable[int] = DEFAULT_DEPTHS,
    episodes_per_depth: int = 10,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """Evaluate the inverse-scramble debugging baseline."""

    _validate_episode_count(episodes_per_depth)
    random_source = rng if rng is not None else random.Random()
    agent = InverseScrambleAgent()
    results: list[dict] = []

    for depth in depths:
        solved_moves: list[int] = []
        solved_count = 0
        start_time = perf_counter()
        for _ in range(episodes_per_depth):
            env = CubeEnvironment(rng=random_source)
            env.scramble(depth)
            solution = agent.solve(env)
            candidate = env.copy()
            for move in solution:
                candidate.cube = apply_move(candidate.cube, move)
            if candidate.is_solved():
                solved_count += 1
                solved_moves.append(len(solution))

        elapsed = perf_counter() - start_time
        results.append(
            _base_metrics(
                agent="inverse_scramble",
                depth=depth,
                episodes=episodes_per_depth,
                solved_count=solved_count,
                elapsed_seconds=elapsed,
                solved_moves=solved_moves,
            )
        )

    return results


def evaluate_bfs_agent(
    depths: Iterable[int] = DEFAULT_DEPTHS,
    episodes_per_depth: int = 1,
    max_depth: int = 7,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """Evaluate shallow BFS and include search-cost metrics."""

    _validate_episode_count(episodes_per_depth)
    random_source = rng if rng is not None else random.Random()
    agent = BFSAgent(max_depth=max_depth)
    results: list[dict] = []

    for depth in depths:
        solved_moves: list[int] = []
        solved_count = 0
        expanded_nodes = 0
        visited_states = 0
        depth_limited_count = 0
        start_time = perf_counter()
        for _ in range(episodes_per_depth):
            env = CubeEnvironment(rng=random_source)
            env.scramble(depth)
            result = agent.solve(env)
            expanded_nodes += result.expanded_nodes
            visited_states += result.visited_states
            depth_limited_count += int(result.depth_limited)
            if result.solved:
                solved_count += 1
                solved_moves.append(result.solution_length)

        elapsed = perf_counter() - start_time
        row = _base_metrics(
            agent="bfs",
            depth=depth,
            episodes=episodes_per_depth,
            solved_count=solved_count,
            elapsed_seconds=elapsed,
            solved_moves=solved_moves,
        )
        row.update(
            {
                "avg_expanded_nodes": expanded_nodes / episodes_per_depth,
                "avg_visited_states": visited_states / episodes_per_depth,
                "depth_limited_count": depth_limited_count,
                "max_search_depth": max_depth,
            }
        )
        results.append(row)

    return results


def write_results_csv(
    results: list[dict], path: Path | str = DEFAULT_CSV_PATH
) -> Path:
    """Write evaluation dictionaries to CSV and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return output_path


def print_results_table(results: list[dict]) -> None:
    """Print a compact console table for baseline metrics."""

    columns = (
        "agent",
        "depth",
        "episodes",
        "solve_rate",
        "avg_solved_moves",
        "avg_time_seconds",
        "avg_expanded_nodes",
        "avg_visited_states",
    )
    print(" | ".join(columns))
    print(" | ".join("-" * len(column) for column in columns))
    for row in results:
        print(
            " | ".join(
                _format_value(row.get(column, "")) for column in columns
            )
        )


def _base_metrics(
    agent: str,
    depth: int,
    episodes: int,
    solved_count: int,
    elapsed_seconds: float,
    solved_moves: list[int],
) -> dict:
    return {
        "agent": agent,
        "depth": depth,
        "episodes": episodes,
        "solved_count": solved_count,
        "solve_rate": solved_count / episodes,
        "avg_solved_moves": (
            sum(solved_moves) / len(solved_moves) if solved_moves else 0.0
        ),
        "avg_time_seconds": elapsed_seconds / episodes,
    }


def _validate_episode_count(episodes_per_depth: int) -> None:
    if episodes_per_depth <= 0:
        raise ValueError("episodes_per_depth must be positive")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

"""Breadth-first search baseline for shallow cube scrambles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from cube.environment import ACTION_TO_MOVE, CubeEnvironment
from cube.moves import apply_move
from cube.notation import Move, parse_move
from cube.state import CubeState


@dataclass(frozen=True)
class SearchResult:
    """Result returned by the BFS baseline."""

    solved: bool
    solution: list[str]
    expanded_nodes: int
    visited_states: int
    elapsed_seconds: float
    max_depth: int
    depth_limited: bool

    @property
    def solution_length(self) -> int:
        return len(self.solution)


class BFSAgent:
    """Shallow BFS solver over the environment's discrete action moves."""

    def __init__(self, max_depth: int = 7, prune_inverse: bool = True) -> None:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        self.max_depth = max_depth
        self.prune_inverse = prune_inverse
        self.actions = tuple(ACTION_TO_MOVE.values())

    def legal_moves(self, previous_move: Optional[str] = None) -> list[str]:
        """Return legal move tokens, optionally pruning immediate inverses."""

        if not self.prune_inverse or previous_move is None:
            return list(self.actions)
        return [
            move
            for move in self.actions
            if not _are_immediate_inverses(previous_move, move)
        ]

    def solve(self, env: CubeEnvironment) -> SearchResult:
        """Search for a solution from the environment's current cube state."""

        start_time = perf_counter()
        start_cube = env.cube.copy()
        start_state = start_cube.to_flat_string()
        if start_cube.is_solved():
            return SearchResult(
                solved=True,
                solution=[],
                expanded_nodes=0,
                visited_states=1,
                elapsed_seconds=perf_counter() - start_time,
                max_depth=self.max_depth,
                depth_limited=False,
            )

        queue = deque([(start_cube, [], None)])
        visited = {start_state}
        expanded_nodes = 0
        depth_limited = False

        while queue:
            cube, path, previous_move = queue.popleft()
            if len(path) >= self.max_depth:
                depth_limited = True
                continue

            expanded_nodes += 1
            for move in self.legal_moves(previous_move):
                next_cube = apply_move(cube, move)
                next_state = next_cube.to_flat_string()
                if next_state in visited:
                    continue

                next_path = [*path, move]
                if next_cube.is_solved():
                    return SearchResult(
                        solved=True,
                        solution=next_path,
                        expanded_nodes=expanded_nodes,
                        visited_states=len(visited) + 1,
                        elapsed_seconds=perf_counter() - start_time,
                        max_depth=self.max_depth,
                        depth_limited=False,
                    )

                visited.add(next_state)
                queue.append((next_cube, next_path, move))

        return SearchResult(
            solved=False,
            solution=[],
            expanded_nodes=expanded_nodes,
            visited_states=len(visited),
            elapsed_seconds=perf_counter() - start_time,
            max_depth=self.max_depth,
            depth_limited=depth_limited,
        )


def _are_immediate_inverses(first: str, second: str) -> bool:
    first_move = _single_quarter_turn(first)
    second_move = _single_quarter_turn(second)
    return (
        first_move.face == second_move.face
        and first_move.wide == second_move.wide
        and first_move.direction == -second_move.direction
    )


def _single_quarter_turn(token: str) -> Move:
    moves = parse_move(token)
    if len(moves) != 1:
        raise ValueError("BFS action moves must be single quarter turns")
    return moves[0]

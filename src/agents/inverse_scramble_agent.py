"""Debugging baseline that solves by reversing the recorded scramble."""

from __future__ import annotations

from cube.environment import CubeEnvironment
from cube.moves import apply_move
from cube.notation import inverse_algorithm, split_algorithm


class InverseScrambleAgent:
    """Use the recorded scramble inverse as a correctness check."""

    def solve(self, env: CubeEnvironment) -> list[str]:
        """Return inverse-scramble moves and verify they solve a copy."""

        solution = split_algorithm(inverse_algorithm(env.scramble_sequence))
        candidate = env.copy()
        for move in solution:
            candidate.cube = apply_move(candidate.cube, move)

        if not candidate.is_solved():
            raise ValueError("inverse scramble did not solve the cube")
        return solution

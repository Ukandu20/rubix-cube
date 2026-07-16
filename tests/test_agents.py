import random
import tempfile
import unittest
from pathlib import Path

from agents.bfs_agent import BFSAgent, SearchResult
from agents.evaluate import (
    evaluate_bfs_agent,
    evaluate_inverse_scramble_agent,
    evaluate_random_agent,
    write_results_csv,
)
from agents.inverse_scramble_agent import InverseScrambleAgent
from agents.random_agent import RandomAgent
from cube.environment import ACTION_SIZE, CubeEnvironment
from cube.moves import apply_algorithm, apply_move


class BaselineAgentTests(unittest.TestCase):
    def test_random_agent_returns_valid_action_indices(self):
        agent = RandomAgent(rng=random.Random(4))
        env = CubeEnvironment()

        for _ in range(50):
            action = agent.act(env)
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, ACTION_SIZE)

    def test_inverse_scramble_agent_solves_known_scramble(self):
        env = CubeEnvironment()
        env.scramble_sequence = "R U F"
        env.cube = apply_algorithm(env.cube, env.scramble_sequence)

        solution = InverseScrambleAgent().solve(env)
        candidate = env.copy()
        for move in solution:
            candidate.cube = apply_move(candidate.cube, move)

        self.assertEqual(solution, ["F'", "U'", "R'"])
        self.assertTrue(candidate.is_solved())
        self.assertFalse(env.is_solved())

    def test_inverse_scramble_agent_solves_generated_scrambles(self):
        env = CubeEnvironment(rng=random.Random(9))
        agent = InverseScrambleAgent()

        for depth in range(1, 6):
            with self.subTest(depth=depth):
                env.scramble(depth)
                solution = agent.solve(env)
                candidate = env.copy()
                for move in solution:
                    candidate.cube = apply_move(candidate.cube, move)

                self.assertEqual(len(solution), depth)
                self.assertTrue(candidate.is_solved())

    def test_bfs_agent_solves_shallow_known_scramble(self):
        env = CubeEnvironment()
        env.cube = apply_algorithm(env.cube, "R U")

        result = BFSAgent(max_depth=4).solve(env)
        candidate = env.copy()
        for move in result.solution:
            candidate.cube = apply_move(candidate.cube, move)

        self.assertIsInstance(result, SearchResult)
        self.assertTrue(result.solved)
        self.assertLessEqual(result.solution_length, 4)
        self.assertGreater(result.expanded_nodes, 0)
        self.assertGreater(result.visited_states, 1)
        self.assertTrue(candidate.is_solved())

    def test_bfs_agent_reports_depth_limit_failure(self):
        env = CubeEnvironment()
        env.cube = apply_algorithm(env.cube, "R U")

        result = BFSAgent(max_depth=1).solve(env)

        self.assertFalse(result.solved)
        self.assertEqual(result.solution, [])
        self.assertTrue(result.depth_limited)

    def test_bfs_prunes_immediate_inverse_moves(self):
        agent = BFSAgent(prune_inverse=True)

        self.assertNotIn("R'", agent.legal_moves("R"))
        self.assertIn("R", agent.legal_moves("R"))
        self.assertIn("U", agent.legal_moves("R"))

    def test_evaluators_return_required_metric_fields(self):
        required = {
            "agent",
            "depth",
            "episodes",
            "solved_count",
            "solve_rate",
            "avg_solved_moves",
            "avg_time_seconds",
        }

        rows = []
        rows.extend(
            evaluate_random_agent(
                depths=(1,), episodes_per_depth=1, rng=random.Random(1)
            )
        )
        rows.extend(
            evaluate_inverse_scramble_agent(
                depths=(1,), episodes_per_depth=1, rng=random.Random(2)
            )
        )
        rows.extend(
            evaluate_bfs_agent(
                depths=(1,), episodes_per_depth=1, max_depth=3, rng=random.Random(3)
            )
        )

        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(required.issubset(row))
        self.assertIn("avg_expanded_nodes", rows[-1])
        self.assertIn("avg_visited_states", rows[-1])

    def test_write_results_csv(self):
        results = [{"agent": "random", "depth": 1, "solve_rate": 0.0}]

        with tempfile.TemporaryDirectory() as directory:
            output_path = write_results_csv(results, Path(directory) / "results.csv")

            self.assertTrue(output_path.exists())
            self.assertIn("random", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

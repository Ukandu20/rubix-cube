import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cube.environment import (
    ACTION_SIZE,
    ACTION_TO_MOVE,
    MOVE_TO_ACTION,
    CubeEnvironment,
)
from cube.notation import is_valid_move, split_algorithm
from cube.state import CubeState


class CubeEnvironmentTests(unittest.TestCase):
    def test_reset_returns_solved_state_and_clears_episode(self):
        env = CubeEnvironment(rng=random.Random(3))
        env.scramble(3)
        env.step(MOVE_TO_ACTION["R"])

        state = env.reset()

        self.assertEqual(state, CubeState.solved().to_flat_string())
        self.assertTrue(env.is_solved())
        self.assertEqual(env.move_count, 0)
        self.assertEqual(env.scramble_depth, 0)
        self.assertEqual(env.scramble_sequence, "")
        self.assertEqual(env.move_history, [])
        self.assertEqual(env.max_steps, 30)

    def test_valid_step_returns_rl_tuple_and_info(self):
        env = CubeEnvironment()

        next_state, reward, done, info = env.step(MOVE_TO_ACTION["R"])

        self.assertIsInstance(next_state, str)
        self.assertEqual(reward, -0.01)
        self.assertFalse(done)
        self.assertEqual(info["move"], "R")
        self.assertEqual(info["move_count"], 1)
        self.assertEqual(info["max_steps"], 30)
        self.assertFalse(info["is_solved"])
        self.assertEqual(info["scramble"], "")
        self.assertFalse(info["timeout"])
        self.assertEqual(env.move_history, ["R"])

    def test_invalid_actions_raise_value_error(self):
        env = CubeEnvironment()

        for action in (-1, ACTION_SIZE, "R"):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    env.step(action)

    def test_inverse_action_solves_cube_and_rewards_success(self):
        env = CubeEnvironment()

        env.step(MOVE_TO_ACTION["R"])
        next_state, reward, done, info = env.step(MOVE_TO_ACTION["R'"])

        self.assertEqual(next_state, CubeState.solved().to_flat_string())
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertTrue(info["is_solved"])
        self.assertFalse(info["timeout"])

    def test_max_episode_length_ends_unsolved_episode(self):
        env = CubeEnvironment(max_steps=2)

        env.step(MOVE_TO_ACTION["R"])
        _, reward, done, info = env.step(MOVE_TO_ACTION["U"])

        self.assertEqual(reward, -0.01)
        self.assertTrue(done)
        self.assertFalse(info["is_solved"])
        self.assertTrue(info["timeout"])

    def test_scramble_changes_cube_and_sets_depth_band_limits(self):
        env = CubeEnvironment(rng=random.Random(12))

        expectations = ((1, 10), (4, 20), (7, 30), (11, 33))
        for depth, max_steps in expectations:
            with self.subTest(depth=depth):
                scramble = env.scramble(depth)
                tokens = split_algorithm(scramble)

                self.assertEqual(len(tokens), depth)
                self.assertTrue(all(is_valid_move(token) for token in tokens))
                self.assertEqual(env.scramble_sequence, scramble)
                self.assertEqual(env.scramble_depth, depth)
                self.assertEqual(env.max_steps, max_steps)
                self.assertEqual(env.move_count, 0)
                self.assertEqual(env.move_history, [])
                self.assertFalse(env.is_solved())

    def test_zero_depth_scramble_resets_without_moves(self):
        env = CubeEnvironment()

        scramble = env.scramble(0)

        self.assertEqual(scramble, "")
        self.assertTrue(env.is_solved())
        self.assertEqual(env.max_steps, 30)

    def test_invalid_scramble_depth_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "depth"):
            CubeEnvironment().scramble(-1)

    def test_get_state_is_solved_and_render(self):
        env = CubeEnvironment()

        self.assertEqual(env.get_state(), CubeState.solved().to_flat_string())
        self.assertTrue(env.is_solved())

        rendered = env.render()
        lines = rendered.splitlines()

        self.assertEqual(len(lines), 9)
        self.assertIn("W W W", lines[0])
        self.assertIn("O O O", rendered)
        self.assertIn("G G G", rendered)
        self.assertIn("R R R", rendered)
        self.assertIn("B B B", rendered)
        self.assertIn("Y Y Y", rendered)

    def test_copy_and_clone_are_independent(self):
        env = CubeEnvironment(rng=random.Random(7))
        env.scramble(3)
        original_state = env.get_state()

        copied = env.copy()
        cloned = env.clone()
        copied.step(MOVE_TO_ACTION["R"])
        cloned.step(MOVE_TO_ACTION["U"])

        self.assertEqual(env.get_state(), original_state)
        self.assertNotEqual(copied.get_state(), env.get_state())
        self.assertNotEqual(cloned.get_state(), env.get_state())
        self.assertEqual(copied.scramble_sequence, env.scramble_sequence)
        self.assertEqual(cloned.scramble_sequence, env.scramble_sequence)

    def test_action_constants_are_consistent(self):
        self.assertEqual(ACTION_SIZE, 12)
        self.assertEqual(ACTION_TO_MOVE[0], "U")
        self.assertEqual(ACTION_TO_MOVE[11], "B'")
        self.assertEqual(MOVE_TO_ACTION["F'"], 5)


if __name__ == "__main__":
    unittest.main()

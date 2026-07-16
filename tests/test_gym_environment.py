import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

import cube as cube_package  # noqa: E402,F401
from cube.environment import MOVE_TO_ACTION
from cube.gym_environment import (
    DEFAULT_EXHAUSTIVE_STATE_THRESHOLD,
    ENV_ID,
    INVERSE_ACTION,
    SOLVED_STATE_STRING,
    RubixCubeSolveEnv,
    decode_state,
    encode_state,
    validate_encoded_state,
)
from cube.moves import apply_move
from cube.state import CubeState


class RubixCubeSolveEnvTests(unittest.TestCase):
    def test_reset_returns_gymnasium_observation_and_info(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = RubixCubeSolveEnv(state_files={1: path}, validate_dataset=True)

            observation, info = env.reset(seed=123)

            self.assertIsInstance(observation, np.ndarray)
            self.assertEqual(observation.dtype, np.int8)
            self.assertEqual(observation.shape, (54,))
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(env.action_space.contains(0))
            self.assertEqual(info["state_source"], "precomputed")
            self.assertEqual(info["start_depth"], 1)
            self.assertEqual(info["sample_id"], "sample_R")
            self.assertEqual(info["scramble_moves"], "R")
            self.assertEqual(info["solution_moves"], "R'")
            self.assertFalse(info["is_solved"])

    def test_reset_can_select_an_exact_state_index(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [_row_for_move("R"), _row_for_move("L")]
            path = _write_depth_csv(directory, 1, rows)
            env = RubixCubeSolveEnv(state_files={1: path})

            observation, info = env.reset(options={"state_index": 1})

            self.assertEqual(info["sample_id"], "sample_L")
            self.assertEqual(encode_state(observation), rows[1]["state_encoded"])

    def test_reset_rejects_out_of_range_state_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = RubixCubeSolveEnv(state_files={1: path})

            for state_index in (-1, 1):
                with self.subTest(state_index=state_index):
                    with self.assertRaisesRegex(ValueError, "state_index"):
                        env.reset(options={"state_index": state_index})

    def test_explicit_state_index_overrides_exhaustive_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [_row_for_move("R"), _row_for_move("L")]
            path = _write_depth_csv(directory, 1, rows)
            env = RubixCubeSolveEnv(
                state_files={1: path},
                exhaustive_state_threshold=DEFAULT_EXHAUSTIVE_STATE_THRESHOLD,
            )

            _, explicit_info = env.reset(options={"state_index": 1})
            _, automatic_info = env.reset()

            self.assertEqual(explicit_info["sample_id"], "sample_L")
            self.assertEqual(automatic_info["sample_id"], "sample_R")

    def test_exhaustive_selection_threshold_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            small_path = _write_depth_csv(
                directory,
                1,
                [_row_for_move("R")] * (DEFAULT_EXHAUSTIVE_STATE_THRESHOLD - 1),
            )
            threshold_path = _write_depth_csv(
                directory,
                2,
                [_row_for_move("U R", depth=2)] * DEFAULT_EXHAUSTIVE_STATE_THRESHOLD,
            )
            env = RubixCubeSolveEnv(
                state_files={1: small_path, 2: threshold_path},
                scramble_depth=None,
                scramble_depth_range=(1, 2),
                validate_dataset=False,
                exhaustive_state_threshold=DEFAULT_EXHAUSTIVE_STATE_THRESHOLD,
            )

            self.assertEqual(env.state_selection_mode(1), "exhaustive_cycle")
            self.assertEqual(env.state_selection_mode(2), "random")

    def test_step_returns_five_values_and_ordinary_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = RubixCubeSolveEnv(state_files={1: path}, max_episode_steps=10)
            env.reset(seed=1)

            observation, reward, terminated, truncated, info = env.step(
                MOVE_TO_ACTION["U"]
            )

            self.assertTrue(env.observation_space.contains(observation))
            self.assertAlmostEqual(reward, -0.01)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["last_move"], "U")
            self.assertEqual(info["move_history_notation"], ["U"])
            self.assertEqual(info["terminated_reason"], "running")

    def test_inverse_solution_solves_depth_one_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("L")])
            env = RubixCubeSolveEnv(state_files={1: path})
            env.reset(seed=2)

            observation, reward, terminated, truncated, info = env.step(
                MOVE_TO_ACTION["L'"]
            )

            self.assertEqual(encode_state(observation), SOLVED_STATE_STRING)
            self.assertAlmostEqual(reward, 0.99)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["is_solved"])
            self.assertEqual(info["terminated_reason"], "solved")

    def test_immediate_inverse_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = RubixCubeSolveEnv(state_files={1: path})
            env.reset(seed=3)

            env.step(MOVE_TO_ACTION["U"])
            _, reward, terminated, truncated, info = env.step(MOVE_TO_ACTION["U'"])

            self.assertAlmostEqual(reward, -0.06)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["immediate_inverse_move"])

    def test_timeout_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = RubixCubeSolveEnv(state_files={1: path}, max_episode_steps=1)
            env.reset(seed=4)

            _, reward, terminated, truncated, info = env.step(MOVE_TO_ACTION["U"])

            self.assertAlmostEqual(reward, -0.11)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["terminated_reason"], "max_steps_reached")

    def test_depth_range_samples_uniform_depth_first(self):
        with tempfile.TemporaryDirectory() as directory:
            depth_1 = _write_depth_csv(directory, 1, [_row_for_move("R")])
            depth_2 = _write_depth_csv(directory, 2, [_row_for_move("U R", depth=2)])
            env = RubixCubeSolveEnv(
                state_files={1: depth_1, 2: depth_2},
                scramble_depth=None,
                scramble_depth_range=(1, 2),
            )

            depths = {env.reset(seed=seed)[1]["start_depth"] for seed in range(20)}

            self.assertEqual(depths, {1, 2})

    def test_parquet_dataset_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_parquet(directory, 1, [_row_for_move("F")])
            env = RubixCubeSolveEnv(state_files={1: path})

            _, info = env.reset(seed=5)

            self.assertEqual(info["scramble_moves"], "F")

    def test_move_followed_by_inverse_and_four_repeats_restore_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("B")])
            env = RubixCubeSolveEnv(state_files={1: path})

            for action, inverse_action in INVERSE_ACTION.items():
                with self.subTest(action=action):
                    env.reset(seed=action)
                    env.cube = CubeState.solved()
                    env.cube_state = decode_state(SOLVED_STATE_STRING)
                    original = env.get_observation()
                    env.step(action)
                    restored, _, _, _, _ = env.step(inverse_action)
                    self.assertEqual(encode_state(restored), encode_state(original))

            for action in range(12):
                with self.subTest(repeat_action=action):
                    env.reset(seed=action)
                    env.cube = CubeState.solved()
                    env.cube_state = decode_state(SOLVED_STATE_STRING)
                    original = env.get_observation()
                    observation = original
                    for _ in range(4):
                        observation, _, terminated, truncated, _ = env.step(action)
                        self.assertFalse(truncated)
                    self.assertTrue(terminated)
                    self.assertEqual(encode_state(observation), encode_state(original))

    def test_render_returns_text_face_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = RubixCubeSolveEnv(state_files={1: path}, render_mode="text")
            env.reset(seed=6)

            rendered = env.render()

            self.assertIsInstance(rendered, str)
            self.assertIn(f"Step: 0 / {env.max_episode_steps}", rendered)
            self.assertIn("Face 0:", rendered)
            self.assertIn("Solved: False", rendered)

    def test_gymnasium_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_depth_csv(directory, 1, [_row_for_move("R")])
            env = gym.make(
                ENV_ID,
                state_files={1: path},
                validate_dataset=True,
            )

            observation, info = env.reset(seed=7)

            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(info["start_depth"], 1)
            env.close()

    def test_state_encoding_helpers_validate_cube_strings(self):
        decoded = decode_state(SOLVED_STATE_STRING)

        self.assertTrue(validate_encoded_state(SOLVED_STATE_STRING))
        self.assertEqual(encode_state(decoded), SOLVED_STATE_STRING)

        with self.assertRaises(ValueError):
            decode_state("Y" * 53)
        with self.assertRaises(ValueError):
            encode_state(np.array([6] * 54, dtype=np.int8))

    def test_validation_rejects_bad_datasets(self):
        bad_rows = {
            "bad length": [{"state_encoded": "Y" * 53}],
            "invalid color": [{"state_encoded": "X" + SOLVED_STATE_STRING[1:]}],
            "bad counts": [{"state_encoded": "Y" * 54}],
            "duplicate": [_row_for_move("R"), _row_for_move("R")],
            "solved state": [{"state_encoded": SOLVED_STATE_STRING}],
        }

        for case, rows in bad_rows.items():
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    path = _write_depth_csv(directory, 1, rows)
                    with self.assertRaises(ValueError):
                        RubixCubeSolveEnv(state_files={1: path})

    def test_validation_rejects_missing_state_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth_1.csv"
            pd.DataFrame([{"wrong": _row_for_move("R")["state_encoded"]}]).to_csv(
                path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "state_encoded"):
                RubixCubeSolveEnv(state_files={1: path})

    def test_validation_rejects_cross_depth_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            row = _row_for_move("R")
            depth_1 = _write_depth_csv(directory, 1, [row])
            depth_2 = _write_depth_csv(directory, 2, [row])

            with self.assertRaisesRegex(ValueError, "already loaded"):
                RubixCubeSolveEnv(state_files={1: depth_1, 2: depth_2})


def _row_for_move(scramble: str, depth: int | None = None) -> dict:
    cube = CubeState.solved()
    for move in scramble.split():
        cube = apply_move(cube, move)
    solution = " ".join(_inverse_token(move) for move in reversed(scramble.split()))
    first_move = solution.split()[0] if solution else ""
    label = scramble.replace(" ", "_").replace("'", "prime")
    return {
        "sample_id": f"sample_{label}",
        "scramble_depth": depth if depth is not None else len(scramble.split()),
        "scramble_moves": scramble,
        "state_encoded": cube.to_flat_string(),
        "solution_moves": solution,
        "first_solution_move": first_move,
    }


def _inverse_token(move: str) -> str:
    return move[:-1] if move.endswith("'") else f"{move}'"


def _write_depth_csv(directory: str, depth: int, rows: list[dict]) -> Path:
    path = Path(directory) / f"depth_{depth}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_depth_parquet(directory: str, depth: int, rows: list[dict]) -> Path:
    path = Path(directory) / f"depth_{depth}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    unittest.main()

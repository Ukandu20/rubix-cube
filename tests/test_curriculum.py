import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.ppo_agent import (  # noqa: E402
    NetworkConfig,
    PPOConfig,
    load_checkpoint,
    train_ppo,
)
from cube.gym_environment import RubixCubeSolveEnv  # noqa: E402
from cube.moves import apply_move  # noqa: E402
from cube.state import CubeState  # noqa: E402
from curriculum.manager import (  # noqa: E402
    AdvancementThreshold,
    CurriculumConfig,
    CurriculumManager,
    load_curriculum_config,
)


class CurriculumManagerTests(unittest.TestCase):
    def test_default_configuration_matches_specification(self):
        config = load_curriculum_config()

        self.assertEqual((config.min_depth, config.max_depth), (1, 5))
        self.assertEqual(config.sampling_strategy, "mixed")
        self.assertEqual(
            config.mixed_sampling_weights[3],
            {1: 0.2, 2: 0.3, 3: 0.5},
        )
        self.assertEqual(
            config.advancement_thresholds[4],
            AdvancementThreshold(0.75, 8.0, 0.15),
        )
        self.assertEqual(config.max_episode_steps(5), 11)

    def test_configuration_rejects_invalid_mixed_weights(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            _config(weights={1: {1: 0.8}})

    def test_seeded_mixed_depth_sampling_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                1: _write_depth_csv(root, 1, [_row_for_moves("R")]),
                2: _write_depth_csv(root, 2, [_row_for_moves("U R")]),
            }
            config = _config(
                max_depth=2,
                starting_depth=2,
                weights={1: {1: 1.0}, 2: {1: 0.3, 2: 0.7}},
            )
            first = CurriculumManager(files, config, seed=7)
            second = CurriculumManager(files, config, seed=7)

            first_depths = [first.sample_state()[0] for _ in range(30)]
            second_depths = [second.sample_state()[0] for _ in range(30)]

        self.assertEqual(first_depths, second_depths)
        self.assertIn(1, first_depths)
        self.assertIn(2, first_depths)

    def test_exhaustive_cycles_are_independent_and_threshold_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth_1_rows = [_row_for_moves("R"), _row_for_moves("L")]
            depth_2_rows = [_row_for_moves("U R"), _row_for_moves("D F")]
            files = {
                1: _write_depth_csv(root, 1, depth_1_rows),
                2: _write_depth_csv(root, 2, depth_2_rows),
            }
            manager = CurriculumManager(
                files,
                _config(max_depth=2),
                exhaustive_state_threshold=500,
            )

            depth_1_ids = [
                manager.sample_state(depth=1)[1]["sample_id"]
                for _ in range(3)
            ]
            depth_2_ids = [
                manager.sample_state(depth=2)[1]["sample_id"]
                for _ in range(3)
            ]

            threshold_files = {
                1: _write_depth_csv(
                    root,
                    1,
                    [_row_for_moves("R")] * 499,
                    name="small.csv",
                ),
                2: _write_depth_csv(
                    root,
                    2,
                    [_row_for_moves("U R")] * 500,
                    name="threshold.csv",
                ),
            }
            threshold_manager = CurriculumManager(
                threshold_files,
                _config(max_depth=2),
                exhaustive_state_threshold=500,
                validate_dataset=False,
            )

        self.assertEqual(depth_1_ids, ["sample_R", "sample_L", "sample_R"])
        self.assertEqual(
            depth_2_ids,
            ["sample_U_R", "sample_D_F", "sample_U_R"],
        )
        self.assertEqual(
            threshold_manager.state_selection_mode(1),
            "exhaustive_cycle",
        )
        self.assertEqual(threshold_manager.state_selection_mode(2), "random")

    def test_advancement_requires_all_thresholds_and_stops_at_max_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = CurriculumManager(
                {
                    1: _write_depth_csv(root, 1, [_row_for_moves("R")]),
                    2: _write_depth_csv(root, 2, [_row_for_moves("U R")]),
                },
                _config(max_depth=2),
            )

            passing = {
                "solve_rate": 0.90,
                "average_solution_length": 2,
                "timeout_rate": 0.10,
            }
            self.assertTrue(manager.should_advance(passing))
            for key, value in (
                ("solve_rate", 0.89),
                ("average_solution_length", 2.1),
                ("timeout_rate", 0.11),
            ):
                failing = dict(passing)
                failing[key] = value
                self.assertFalse(manager.should_advance(failing))

            self.assertTrue(manager.increase_depth(timestep=100, metrics=passing))
            self.assertEqual(manager.get_current_depth(), 2)
            self.assertFalse(manager.should_advance(passing))
            self.assertFalse(manager.increase_depth())
            self.assertEqual(manager.events[0]["timestep"], 100)

    def test_environment_records_curriculum_and_uses_level_episode_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                1: _write_depth_csv(root, 1, [_row_for_moves("R")]),
                2: _write_depth_csv(root, 2, [_row_for_moves("U R")]),
            }
            config = _config(
                max_depth=2,
                starting_depth=2,
                weights={1: {1: 1.0}, 2: {1: 1.0, 2: 0.0}},
            )
            manager = CurriculumManager(files, config, seed=1)
            env = RubixCubeSolveEnv(
                curriculum_manager=manager,
                max_episode_steps=50,
            )

            _, info = env.reset()

        self.assertEqual(info["curriculum_depth"], 2)
        self.assertEqual(info["sampled_depth"], 1)
        self.assertEqual(info["max_episode_steps"], 5)
        self.assertEqual(env.episode_max_steps, 5)

    def test_training_persists_curriculum_configuration_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_dir = root / "output"
            _write_depth_csv(data_dir, 1, [_row_for_moves("R")])
            config = _config()

            result = train_ppo(
                total_timesteps=2,
                data_dir=data_dir,
                output_dir=output_dir,
                eval_frequency=2,
                eval_episodes=1,
                config=PPOConfig(n_steps=2, n_epochs=1, batch_size=2),
                network_config=NetworkConfig(hidden_layers=(8,)),
                curriculum_config=config,
            )
            run_config = json.loads(
                (output_dir / "config.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (output_dir / "curriculum_progress.json").read_text(
                    encoding="utf-8"
                )
            )
            best_checkpoint = load_checkpoint(
                output_dir / "best_model.pt",
                device="cpu",
            )[1]
            depth_checkpoint = load_checkpoint(
                output_dir / "best_model_depth_1.pt",
                device="cpu",
            )[1]

        self.assertEqual(run_config["curriculum"]["sampling_strategy"], "mixed")
        self.assertEqual(
            run_config["training_session"],
            metrics["training_session"],
        )
        self.assertTrue(run_config["training_session"]["started_at"])
        self.assertTrue(run_config["training_session"]["completed_at"])
        self.assertGreaterEqual(
            run_config["training_session"]["elapsed_seconds"],
            0.0,
        )
        self.assertEqual(progress["current_depth"], 1)
        self.assertEqual(len(progress["evaluations"]), 1)
        self.assertIn("curriculum_progress", metrics)
        self.assertEqual(
            metrics["checkpoint_selection"]["strategy"],
            "curriculum_lexicographic",
        )
        self.assertEqual(
            metrics["checkpoint_selection"]["best_curriculum_depth"],
            1,
        )
        self.assertEqual(
            best_checkpoint["metadata"]["selection"],
            "curriculum_lexicographic",
        )
        self.assertEqual(
            depth_checkpoint["metadata"]["selection"],
            "best_at_curriculum_depth",
        )
        self.assertEqual(
            result["metrics"]["history"][0]["curriculum_evaluation"][
                "curriculum_depth"
            ],
            1,
        )

    def test_training_saves_latest_passed_curriculum_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_dir = root / "output"
            _write_depth_csv(data_dir, 1, [_row_for_moves("R")])
            _write_depth_csv(data_dir, 2, [_row_for_moves("U R")])

            with patch.object(
                CurriculumManager,
                "should_advance",
                return_value=True,
            ):
                train_ppo(
                    total_timesteps=2,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    eval_frequency=2,
                    eval_episodes=1,
                    config=PPOConfig(n_steps=2, n_epochs=1, batch_size=2),
                    network_config=NetworkConfig(hidden_layers=(8,)),
                    curriculum_config=_config(max_depth=2),
                )

            metadata = load_checkpoint(
                output_dir / "latest_passed_gate.pt",
                device="cpu",
            )[1]["metadata"]

        self.assertEqual(metadata["selection"], "latest_passed_curriculum_gate")
        self.assertEqual(metadata["passed_curriculum_depth"], 1)


def _config(
    *,
    max_depth: int = 1,
    starting_depth: int = 1,
    weights: dict[int, dict[int, float]] | None = None,
) -> CurriculumConfig:
    default_weights = {
        depth: {depth: 1.0}
        for depth in range(1, max_depth + 1)
    }
    return CurriculumConfig(
        min_depth=1,
        max_depth=max_depth,
        starting_depth=starting_depth,
        sampling_strategy="mixed",
        mixed_sampling_weights=weights or default_weights,
        advancement_thresholds={
            depth: AdvancementThreshold(
                success_rate=0.9 if depth == 1 else 0.85,
                max_average_moves=2 * depth,
                max_timeout_rate=0.1,
            )
            for depth in range(1, max_depth + 1)
        },
    )


def _row_for_moves(moves: str) -> dict:
    cube = CubeState.solved()
    for move in moves.split():
        cube = apply_move(cube, move)
    return {
        "sample_id": f"sample_{moves.replace(' ', '_')}",
        "scramble_depth": len(moves.split()),
        "scramble_moves": moves,
        "state_encoded": cube.to_flat_string(),
    }


def _write_depth_csv(
    directory: Path,
    depth: int,
    rows: list[dict],
    *,
    name: str | None = None,
) -> Path:
    path = directory / (name or f"depth_{depth}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    unittest.main()

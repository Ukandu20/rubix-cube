import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.sb3_ppo_agent import OneHotObservation, SB3PPOConfig, train_sb3_ppo
from agents.sb3_supervised_warm_start import (
    ACTION_ORDER,
    SupervisedWarmStartConfig,
    WarmStartExample,
    depth_sampling_weights,
    encode_sb3_state,
    epoch_indices,
    load_supervised_warm_start_checkpoint,
    prepare_warm_start_data,
    run_supervised_warm_start,
)
from cube.gym_environment import SOLVED_STATE_STRING, decode_state
from curriculum.manager import AdvancementThreshold, CurriculumConfig
from data.training_data import FIELDNAMES, generate_depth_dataset


class StaticCubeEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=5, shape=(54,), dtype=np.int8)
    action_space = gym.spaces.Discrete(12)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return decode_state(SOLVED_STATE_STRING), {}

    def step(self, action):
        return decode_state(SOLVED_STATE_STRING), 0.0, True, False, {}


def curriculum(max_depth=2):
    return CurriculumConfig(
        min_depth=1,
        max_depth=max_depth,
        starting_depth=1,
        sampling_strategy="mixed",
        mixed_sampling_weights={
            depth: {candidate: 1 / depth for candidate in range(1, depth + 1)}
            for depth in range(1, max_depth + 1)
        },
        advancement_thresholds={
            depth: AdvancementThreshold(0.5, depth * 2, 0.5)
            for depth in range(1, max_depth + 1)
        },
    )


def make_model():
    env = DummyVecEnv([lambda: OneHotObservation(StaticCubeEnv())])
    return PPO(
        "MlpPolicy",
        env,
        n_steps=2,
        batch_size=2,
        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
        seed=3,
        device="cpu",
    )


class SupervisedWarmStartTests(unittest.TestCase):
    def test_depth_modes_resolve_exact_ranges(self):
        cfg = curriculum(10)
        self.assertEqual(
            SupervisedWarmStartConfig(depth_mode="mastered", max_depth=6).selected_depths(cfg),
            tuple(range(1, 6)),
        )
        self.assertEqual(
            SupervisedWarmStartConfig(depth_mode="frontier", max_depth=6).selected_depths(cfg),
            tuple(range(1, 7)),
        )
        self.assertEqual(
            SupervisedWarmStartConfig(depth_mode="full-curriculum").selected_depths(cfg),
            tuple(range(1, 11)),
        )
        self.assertEqual(
            SupervisedWarmStartConfig(depth_mode="custom", min_depth=3, max_depth=5).selected_depths(cfg),
            (3, 4, 5),
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "required"):
            SupervisedWarmStartConfig(depth_mode="frontier")
        with self.assertRaisesRegex(ValueError, "shared encoder"):
            SupervisedWarmStartConfig(
                depth_mode="frontier", max_depth=1, update_shared_encoder=True
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            train_sb3_ppo(
                curriculum_config=curriculum(1),
                resume_from="checkpoint.zip",
                supervised_config=SupervisedWarmStartConfig(
                    depth_mode="frontier", max_depth=1
                ),
            )

    def test_sb3_encoder_uses_environment_color_order(self):
        encoded = encode_sb3_state(SOLVED_STATE_STRING)
        self.assertEqual(tuple(encoded.shape), (324,))
        self.assertEqual(float(encoded.sum()), 54.0)
        self.assertEqual(int(encoded[:6].argmax()), 0)  # Y is environment index 0.

    def test_preparation_is_deterministic_and_partitions_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            rows = [example.to_row() for example in generate_depth_dataset(1, 12, unique=True)]
            rows.append({**rows[0], "sample_id": "duplicate", "first_solution_move": "U"})
            self._write_rows(data / "depth_1.csv", rows)
            config = SupervisedWarmStartConfig(
                depth_mode="frontier", max_depth=1, sample_per_depth=12
            )
            first = prepare_warm_start_data(
                config=config, curriculum=curriculum(1), data_dir=data,
                output_dir=root / "first", seed=42,
            )
            second = prepare_warm_start_data(
                config=config, curriculum=curriculum(1), data_dir=data,
                output_dir=root / "second", seed=42,
            )

            first_hashes = {
                split: [example.state_hash for example in examples]
                for split, examples in first.splits.items()
            }
            second_hashes = {
                split: [example.state_hash for example in examples]
                for split, examples in second.splits.items()
            }
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(sum(map(len, first.splits.values())), 12)
            self.assertFalse(
                {example.state for example in first.splits["train"]}
                & {example.state for example in first.splits["test"]}
            )
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_counts"], {"1": 12})
            aggregated = [
                row for row in manifest["selected_rows"]["1"]
                if "duplicate" in row["sample_ids"]
            ]
            self.assertEqual(len(aggregated), 1)
            capped = prepare_warm_start_data(
                config=SupervisedWarmStartConfig(
                    depth_mode="frontier", max_depth=1, sample_per_depth=9
                ),
                curriculum=curriculum(1), data_dir=data,
                output_dir=root / "capped", seed=42,
            )
            self.assertEqual(sum(map(len, capped.splits.values())), 9)

    def test_depth_samplers_follow_declared_probabilities(self):
        examples = [
            WarmStartExample(str(index), 1, (0,), (str(index),))
            for index in range(10)
        ] + [
            WarmStartExample(str(index), 2, (0,), (str(index),))
            for index in range(10, 40)
        ]
        balanced = SupervisedWarmStartConfig(
            depth_mode="frontier", max_depth=2, depth_sampling="balanced"
        )
        natural = SupervisedWarmStartConfig(
            depth_mode="frontier", max_depth=2, depth_sampling="natural"
        )
        weighted = SupervisedWarmStartConfig(
            depth_mode="frontier", max_depth=2, depth_sampling="frontier-weighted"
        )
        self.assertEqual(depth_sampling_weights(balanced, curriculum(2), (1, 2), examples), {1: 0.5, 2: 0.5})
        self.assertEqual(depth_sampling_weights(natural, curriculum(2), (1, 2), examples), {1: 0.25, 2: 0.75})
        self.assertEqual(depth_sampling_weights(weighted, curriculum(2), (1, 2), examples), {1: 0.5, 2: 0.5})
        indices = epoch_indices(
            examples,
            weights={1: 0.0, 2: 1.0},
            strategy="frontier-weighted",
            seed=1,
        )
        self.assertTrue(all(examples[index].depth == 2 for index in indices))

    def test_actor_only_training_saves_compatible_best_and_last_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            rows = [example.to_row() for example in generate_depth_dataset(1, 12, unique=True)]
            self._write_rows(data / "depth_1.csv", rows)
            model = make_model()
            critic_before = {
                name: parameter.detach().clone()
                for name, parameter in model.policy.named_parameters()
                if "value_net" in name
            }
            result = run_supervised_warm_start(
                model,
                config=SupervisedWarmStartConfig(
                    depth_mode="frontier", max_depth=1, epochs=1,
                    batch_size=4, sample_per_depth=12,
                    rollout_sample_per_depth=1,
                ),
                curriculum=curriculum(1), data_dir=data,
                output_dir=output, seed=4,
            )

            self.assertTrue((output / "supervised_warm_start_best.pt").exists())
            self.assertTrue((output / "supervised_warm_start_last.pt").exists())
            self.assertEqual(result["provenance"]["critic_head_updated"], False)
            for name, parameter in model.policy.named_parameters():
                if name in critic_before:
                    self.assertTrue(torch.equal(parameter, critic_before[name]))
            payload = load_supervised_warm_start_checkpoint(
                model, output / "supervised_warm_start_best.pt"
            )
            self.assertEqual(tuple(payload["action_schema"]["action_order"]), ACTION_ORDER)
            payload["action_schema"]["action_order"] = list(reversed(ACTION_ORDER))
            incompatible = output / "incompatible.pt"
            torch.save(payload, incompatible)
            with self.assertRaisesRegex(ValueError, "action order mismatch"):
                load_supervised_warm_start_checkpoint(model, incompatible)

    def test_tiny_warm_started_ppo_run_retains_provenance_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            rows = [example.to_row() for example in generate_depth_dataset(1, 12, unique=True)]
            self._write_rows(data / "depth_1.csv", rows)

            result = train_sb3_ppo(
                curriculum_config=curriculum(1),
                total_timesteps=2,
                data_dir=data,
                output_dir=output,
                eval_frequency=2,
                eval_episodes=1,
                config=SB3PPOConfig(
                    n_steps=2, batch_size=2, n_epochs=1, hidden_layers=(16,)
                ),
                supervised_config=SupervisedWarmStartConfig(
                    depth_mode="frontier", max_depth=1, epochs=1,
                    batch_size=4, sample_per_depth=9,
                    rollout_sample_per_depth=1,
                ),
                seed=5,
                device="cpu",
            )

            self.assertEqual(result["config"]["actual_timesteps"], 2)
            self.assertEqual(
                result["config"]["initialization_mode"],
                "supervised_actor_warm_start",
            )
            phases = {
                row["phase"] for row in result["metrics"]["transition_diagnostics"]
            }
            self.assertIn("after_first_ppo_rollout", phases)
            self.assertIn("after_first_ppo_update", phases)
            sidecar = json.loads(
                (output / "final_model.metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sidecar["initialization_mode"], "supervised_actor_warm_start"
            )

    def _write_rows(self, path, rows):
        fieldnames = list(FIELDNAMES)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

from agents.sb3_ppo_agent import (
    OneHotObservation,
    SB3RawObservationPolicy,
    _restore_curriculum_progress,
    build_curriculum_callback,
)
from cube.gym_environment import SOLVED_STATE_STRING, decode_state
from curriculum.manager import AdvancementThreshold
from showcase.service import discover_checkpoints


class StaticCubeEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=5, shape=(54,), dtype=np.int8)
    action_space = gym.spaces.Discrete(12)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return decode_state(SOLVED_STATE_STRING), {}

    def step(self, action):
        return decode_state(SOLVED_STATE_STRING), 0.0, True, False, {}


class SB3PPOAgentTests(unittest.TestCase):
    def test_one_hot_wrapper_exposes_324_categorical_features(self):
        env = OneHotObservation(StaticCubeEnv())
        observation, _ = env.reset()

        self.assertEqual(observation.shape, (324,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(float(observation.sum()), 54.0)
        self.assertTrue(env.observation_space.contains(observation))

    def test_raw_policy_adapter_one_hots_before_prediction(self):
        model = Mock()
        model.predict.return_value = (np.array([7]), None)
        policy = SB3RawObservationPolicy(model)

        action, _ = policy.predict(decode_state(SOLVED_STATE_STRING))

        self.assertEqual(int(action), 7)
        encoded = model.predict.call_args.args[0]
        self.assertEqual(encoded.shape, (1, 324))
        self.assertEqual(float(encoded.sum()), 54.0)
        model.predict.assert_called_once()

    def test_wrapper_rejects_wrong_observation_shape(self):
        env = OneHotObservation(StaticCubeEnv())
        with self.assertRaisesRegex(ValueError, "54 cube stickers"):
            env.observation(np.zeros(53, dtype=np.int8))

    def test_showcase_discovers_sb3_checkpoint_from_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sb3_ppo" / "depth_1_onehot" / "v001" / "final_model.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"placeholder")
            archive.with_suffix(".metadata.json").write_text(
                json.dumps(
                    {
                        "trainer": "stable_baselines3",
                        "timesteps": 100,
                        "evaluation": {"by_depth": {"1": {}}},
                        "curriculum": {"current_depth": 1},
                    }
                ),
                encoding="utf-8",
            )

            checkpoints = discover_checkpoints(root)

            self.assertEqual(len(checkpoints), 1)
            self.assertTrue(checkpoints[0].compatible)
            self.assertEqual(checkpoints[0].trainer, "stable_baselines3")
            self.assertEqual(checkpoints[0].validated_depth, 1)

    @patch("agents.sb3_ppo_agent._save_sb3_checkpoint")
    @patch("agents.sb3_ppo_agent.evaluate_ppo_model")
    def test_curriculum_callback_synchronizes_all_training_envs(
        self, evaluate, save_checkpoint
    ):
        metrics = {
            "solve_rate": 1.0,
            "timeout_rate": 0.0,
            "average_solution_length": 1.0,
        }
        evaluate.return_value = {
            "by_depth": {"1": metrics},
            "overall": metrics,
        }
        manager = Mock()
        current_depth = [1]
        manager.get_current_depth.side_effect = lambda: current_depth[0]
        manager.should_advance.return_value = True

        def advance(**kwargs):
            current_depth[0] = 2
            return True

        manager.increase_depth.side_effect = advance
        manager.progress.side_effect = lambda: {
            "current_depth": current_depth[0],
            "events": [],
        }
        manager.config.max_episode_steps.return_value = 3
        manager.config.advancement_thresholds = {1: AdvancementThreshold(0.9, 2, 0.1)}
        with tempfile.TemporaryDirectory() as directory:
            callback = build_curriculum_callback(
                manager=manager,
                output_dir=Path(directory),
                eval_frequency=10,
                eval_episodes=1,
                data_dir=Path(directory),
                validate_dataset=False,
            )
            training_env = Mock()
            callback.model = Mock()
            callback.model.get_env.return_value = training_env
            callback.num_timesteps = 10

            self.assertTrue(callback._on_step())

            training_env.env_method.assert_called_once_with("set_curriculum_depth", 2)
            self.assertEqual(callback.evaluations[0]["curriculum_depth"], 1)
            self.assertTrue(callback.evaluations[0]["advanced_curriculum"])
            self.assertEqual(save_checkpoint.call_count, 3)

    def test_resume_restores_curriculum_and_checkpoint_selection(self):
        manager = Mock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "curriculum_progress.json").write_text(
                json.dumps(
                    {
                        "current_depth": 3,
                        "events": [{"from_depth": 2, "to_depth": 3}],
                        "evaluations": [{"timesteps": 100}],
                    }
                ),
                encoding="utf-8",
            )
            (output / "metrics.json").write_text(
                json.dumps(
                    {
                        "checkpoint_selection": {
                            "best_curriculum_depth": 3,
                            "best_solve_rate": 0.75,
                        }
                    }
                ),
                encoding="utf-8",
            )

            evaluations, selection = _restore_curriculum_progress(output, manager)

        manager.set_depth.assert_called_once_with(3)
        self.assertEqual(manager.events, [{"from_depth": 2, "to_depth": 3}])
        self.assertEqual(evaluations, [{"timesteps": 100}])
        self.assertEqual(selection["best_curriculum_depth"], 3)


if __name__ == "__main__":
    unittest.main()

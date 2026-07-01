import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.behavior_logging import BehaviorLogConfig, BehaviorLogger
from agents.ppo_agent import (  # noqa: E402
    ActorCriticNet,
    EXHAUSTIVE_EVAL_STATE_THRESHOLD,
    NetworkConfig,
    OBSERVATION_SIZE,
    ONE_HOT_OBSERVATION_SIZE,
    PPOConfig,
    RolloutBuffer,
    evaluate_ppo_model,
    evaluate_random_baseline,
    load_checkpoint,
    make_training_env,
    ppo_update,
    predict_action,
    preprocess_observations,
    save_checkpoint,
    select_action,
    _evaluation_selection_mode,
    _episode_window_metrics,
)
from cube.gym_environment import SOLVED_STATE_STRING, decode_state, encode_state
from cube.moves import apply_move
from cube.state import CubeState


class AlwaysRPrimePolicy:
    def predict(self, observation, deterministic=True):
        return 3, None


class RecordingPolicy:
    def __init__(self):
        self.start_states = []

    def predict(self, observation, deterministic=True):
        self.start_states.append(encode_state(observation))
        return 0, None


class PPOAgentTests(unittest.TestCase):
    def test_empty_episode_window_metrics_include_behavior_rates(self):
        metrics = _episode_window_metrics(deque())

        self.assertEqual(metrics["timeout_rate"], 0.0)
        self.assertEqual(metrics["inverse_move_rate"], 0.0)

    def test_episode_window_metrics_calculate_behavior_rates(self):
        episodes = deque(
            [
                {
                    "reward": 1.0,
                    "length": 2,
                    "solved": True,
                    "timeout": False,
                    "inverse_moves": 1,
                },
                {
                    "reward": -1.0,
                    "length": 4,
                    "solved": False,
                    "timeout": True,
                    "inverse_moves": 2,
                },
            ]
        )

        metrics = _episode_window_metrics(episodes)

        self.assertEqual(metrics["timeout_rate"], 0.5)
        self.assertEqual(metrics["inverse_move_rate"], 0.5)

    def test_network_forward_shapes(self):
        model = ActorCriticNet(NetworkConfig(hidden_layers=(32, 16)))
        observations = torch.zeros((4, ONE_HOT_OBSERVATION_SIZE), dtype=torch.float32)

        logits, values = model(observations)

        self.assertEqual(tuple(logits.shape), (4, 12))
        self.assertEqual(tuple(values.shape), (4,))

    def test_normalized_network_forward_shapes(self):
        model = ActorCriticNet(
            NetworkConfig(
                input_dim=OBSERVATION_SIZE,
                hidden_layers=(32,),
                observation_encoding="normalized",
            )
        )
        observations = torch.zeros((4, OBSERVATION_SIZE), dtype=torch.float32)

        logits, values = model(observations)

        self.assertEqual(tuple(logits.shape), (4, 12))
        self.assertEqual(tuple(values.shape), (4,))

    def test_actor_and_critic_have_independent_parameters_and_gradients(self):
        model = ActorCriticNet(NetworkConfig(hidden_layers=(16,)))
        observations = torch.zeros((4, ONE_HOT_OBSERVATION_SIZE))
        actor_parameter_ids = {id(parameter) for parameter in model.actor_parameters()}
        critic_parameter_ids = {id(parameter) for parameter in model.critic_parameters()}

        self.assertTrue(actor_parameter_ids)
        self.assertTrue(critic_parameter_ids)
        self.assertTrue(actor_parameter_ids.isdisjoint(critic_parameter_ids))

        logits, _ = model(observations)
        logits.sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.actor_parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.critic_parameters())
        )

        model.zero_grad()
        _, values = model(observations)
        values.sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.actor_parameters())
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.critic_parameters())
        )

    def test_action_selection_and_prediction_return_valid_actions(self):
        model = ActorCriticNet(NetworkConfig(hidden_layers=(32,)))
        observation = decode_state(SOLVED_STATE_STRING)

        action, log_prob, value = select_action(model, observation)
        greedy_action = predict_action(model, observation)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 12)
        self.assertGreaterEqual(greedy_action, 0)
        self.assertLess(greedy_action, 12)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)

    def test_preprocess_one_hot_observations_by_default(self):
        observation = np.array([0, 5] * 27, dtype=np.int8)

        tensor = preprocess_observations(observation)

        self.assertEqual(tuple(tensor.shape), (1, ONE_HOT_OBSERVATION_SIZE))
        self.assertEqual(float(tensor.sum().item()), 54.0)
        self.assertEqual(set(tensor.unique().tolist()), {0.0, 1.0})

    def test_preprocess_normalizes_observations_when_requested(self):
        observation = np.array([0, 5] * 27, dtype=np.int8)

        tensor = preprocess_observations(
            observation,
            observation_encoding="normalized",
        )

        self.assertEqual(tuple(tensor.shape), (1, 54))
        self.assertEqual(float(tensor.min().item()), 0.0)
        self.assertEqual(float(tensor.max().item()), 1.0)

    def test_gae_returns_match_hand_computed_values(self):
        buffer = RolloutBuffer()
        observation = decode_state(SOLVED_STATE_STRING)
        buffer.add(
            observation=observation,
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.1,
            value=0.5,
            depth=1,
            info={},
        )
        buffer.add(
            observation=observation,
            action=1,
            reward=1.0,
            done=True,
            log_prob=-0.2,
            value=0.25,
            depth=1,
            info={},
        )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            last_done=True,
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertTrue(torch.allclose(advantages, torch.tensor([1.5, 0.75])))
        self.assertTrue(torch.allclose(returns, torch.tensor([2.0, 1.0])))

    def test_ppo_update_changes_model_parameters(self):
        torch.manual_seed(0)
        model = ActorCriticNet(NetworkConfig(hidden_layers=(16,)))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        buffer = RolloutBuffer()
        observation = decode_state(SOLVED_STATE_STRING)
        for index in range(4):
            buffer.add(
                observation=observation,
                action=index % 2,
                reward=1.0,
                done=index == 3,
                log_prob=-1.0,
                value=0.0,
                depth=1,
                info={},
            )
        buffer.compute_returns_and_advantages(
            last_value=0.0,
            last_done=True,
            gamma=0.99,
            gae_lambda=0.95,
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]

        with patch("agents.ppo_agent.nn.utils.clip_grad_norm_") as clip_grad_norm:
            metrics = ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                config=PPOConfig(n_epochs=2, batch_size=2, target_kl=None),
                device=torch.device("cpu"),
            )

        changed = any(
            not torch.allclose(previous, current)
            for previous, current in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        self.assertIn("policy_loss", metrics)
        self.assertIn("explained_variance", metrics)
        self.assertEqual(clip_grad_norm.call_count, 8)
        actor_parameter_ids = {id(parameter) for parameter in model.actor_parameters()}
        critic_parameter_ids = {id(parameter) for parameter in model.critic_parameters()}
        for call_index, call in enumerate(clip_grad_norm.call_args_list):
            clipped_parameter_ids = {id(parameter) for parameter in call.args[0]}
            expected_ids = (
                actor_parameter_ids
                if call_index % 2 == 0
                else critic_parameter_ids
            )
            self.assertEqual(clipped_parameter_ids, expected_ids)

    def test_deterministic_evaluation_reports_per_depth_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_depth_csv(data_dir, 1, [_row_for_moves("R")])
            _write_depth_csv(data_dir, 2, [_row_for_moves("R U", depth=2)])

            results = evaluate_ppo_model(
                AlwaysRPrimePolicy(),
                depths=(1,),
                episodes_per_depth=2,
                max_episode_steps=3,
                data_dir=data_dir,
            )

        self.assertEqual(results["episodes_per_depth"], 2)
        self.assertEqual(results["by_depth"]["1"]["episodes"], 1)
        self.assertEqual(results["by_depth"]["1"]["solve_rate"], 1.0)
        self.assertEqual(results["overall"]["solved_count"], 1)
        self.assertEqual(results["by_depth"]["1"]["action_counts"]["R'"], 1)
        self.assertEqual(results["by_depth"]["1"]["action_distribution"]["R'"], 1.0)
        self.assertEqual(results["by_depth"]["1"]["available_states"], 1)
        self.assertEqual(
            results["by_depth"]["1"]["state_selection"],
            "exhaustive_cycle",
        )

    def test_small_evaluation_dataset_evaluates_each_state_once_in_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            rows = [
                _row_for_moves("R"),
                _row_for_moves("L"),
                _row_for_moves("F"),
            ]
            _write_depth_csv(data_dir, 1, rows)
            policy = RecordingPolicy()

            results = evaluate_ppo_model(
                policy,
                depths=(1,),
                episodes_per_depth=8,
                max_episode_steps=1,
                data_dir=data_dir,
            )

        expected_cycle = [row["state_encoded"] for row in rows]
        self.assertEqual(policy.start_states, expected_cycle)
        self.assertEqual(results["episodes_per_depth"], 8)
        self.assertEqual(results["by_depth"]["1"]["episodes"], 3)
        self.assertEqual(results["by_depth"]["1"]["available_states"], 3)
        self.assertEqual(
            results["by_depth"]["1"]["state_selection"],
            "exhaustive_cycle",
        )

    def test_small_evaluation_respects_requested_count_below_available_states(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            rows = [
                _row_for_moves("R"),
                _row_for_moves("L"),
                _row_for_moves("F"),
            ]
            _write_depth_csv(data_dir, 1, rows)
            policy = RecordingPolicy()

            results = evaluate_ppo_model(
                policy,
                depths=(1,),
                episodes_per_depth=2,
                max_episode_steps=1,
                data_dir=data_dir,
            )

        self.assertEqual(
            policy.start_states,
            [row["state_encoded"] for row in rows[:2]],
        )
        self.assertEqual(results["episodes_per_depth"], 2)
        self.assertEqual(results["by_depth"]["1"]["episodes"], 2)

    def test_evaluation_selection_threshold_is_strict(self):
        self.assertEqual(
            _evaluation_selection_mode(EXHAUSTIVE_EVAL_STATE_THRESHOLD - 1),
            "exhaustive_cycle",
        )
        self.assertEqual(
            _evaluation_selection_mode(EXHAUSTIVE_EVAL_STATE_THRESHOLD),
            "random",
        )

    def test_small_training_dataset_cycles_evenly_in_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            rows = [
                _row_for_moves("R"),
                _row_for_moves("L"),
                _row_for_moves("F"),
            ]
            _write_depth_csv(data_dir, 1, rows)
            env = make_training_env(
                min_depth=1,
                max_depth=1,
                data_dir=data_dir,
            )

            selected_states = [encode_state(env.reset()[0]) for _ in range(8)]

        expected_cycle = [row["state_encoded"] for row in rows]
        self.assertEqual(selected_states, (expected_cycle * 3)[:8])

    def test_training_cycles_are_independent_per_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            depth_1_rows = [
                _row_for_moves("R"),
                _row_for_moves("L"),
            ]
            depth_2_rows = [
                _row_for_moves("U R", depth=2),
                _row_for_moves("D F", depth=2),
            ]
            _write_depth_csv(data_dir, 1, depth_1_rows)
            _write_depth_csv(data_dir, 2, depth_2_rows)
            env = make_training_env(
                min_depth=1,
                max_depth=2,
                data_dir=data_dir,
            )

            selections = {
                depth: [
                    encode_state(env.reset(options={"scramble_depth": depth})[0])
                    for _ in range(3)
                ]
                for depth in (1, 2)
            }

        self.assertEqual(
            selections[1],
            [
                depth_1_rows[0]["state_encoded"],
                depth_1_rows[1]["state_encoded"],
                depth_1_rows[0]["state_encoded"],
            ],
        )
        self.assertEqual(
            selections[2],
            [
                depth_2_rows[0]["state_encoded"],
                depth_2_rows[1]["state_encoded"],
                depth_2_rows[0]["state_encoded"],
            ],
        )

    def test_random_baseline_reports_expected_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_depth_csv(data_dir, 1, [_row_for_moves("R")])

            results = evaluate_random_baseline(
                depths=(1,),
                episodes_per_depth=3,
                max_episode_steps=2,
                data_dir=data_dir,
                seed=3,
            )
            repeated_results = evaluate_random_baseline(
                depths=(1,),
                episodes_per_depth=3,
                max_episode_steps=2,
                data_dir=data_dir,
                seed=3,
            )

        self.assertIn("1", results["by_depth"])
        self.assertEqual(results, repeated_results)
        self.assertEqual(results["by_depth"]["1"]["episodes"], 3)
        self.assertIn("solve_rate", results["overall"])
        self.assertIn("inverse_move_rate", results["overall"])
        self.assertIn("action_distribution", results["overall"])
        self.assertAlmostEqual(
            sum(results["overall"]["action_distribution"].values()),
            1.0,
        )

    def test_behavior_logging_writes_episode_and_step_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            logs_root = root / "data" / "logs"
            _write_depth_csv(data_dir, 1, [_row_for_moves("R")])
            logger = BehaviorLogger(
                BehaviorLogConfig(
                    logs_root=logs_root,
                    model_version="ppo_test",
                    run_id="run_001",
                )
            )

            evaluate_ppo_model(
                ActorCriticNet(NetworkConfig(hidden_layers=(16,))),
                depths=(1,),
                episodes_per_depth=1,
                max_episode_steps=1,
                data_dir=data_dir,
                behavior_logger=logger,
            )
            logger.flush()

            episode_path = logs_root / "ppo_test" / "run_001" / "depth_1" / "episodes.parquet"
            step_path = logs_root / "ppo_test" / "run_001" / "depth_1" / "steps.parquet"
            episodes = pd.read_parquet(episode_path)
            steps = pd.read_parquet(step_path)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(steps), 1)
        self.assertTrue(
            {
                "episode_id",
                "model_version",
                "scramble_depth",
                "scramble_sequence",
                "start_state",
                "solved",
                "total_steps",
                "total_reward",
                "timeout",
                "termination_reason",
                "moves_taken",
                "moves_taken_count",
                "seed",
                "timestamp",
            }.issubset(episodes.columns)
        )
        self.assertTrue(
            {
                "episode_id",
                "step_index",
                "start_state",
                "action",
                "move",
                "end_state",
                "reward",
                "total_reward_so_far",
                "previous_action",
                "immediate_inverse_move",
                "solved_after_move",
                "done",
                "termination_reason",
                "action_probability",
                "log_probability",
                "value_estimate",
                "entropy",
            }.issubset(steps.columns)
        )
        self.assertEqual(episodes.loc[0, "model_version"], "ppo_test")
        self.assertTrue(pd.isna(episodes.loc[0, "seed"]))
        self.assertEqual(episodes.loc[0, "moves_taken"], steps.loc[0, "move"])
        self.assertEqual(int(episodes.loc[0, "moves_taken_count"]), 1)
        self.assertEqual(len(steps.loc[0, "start_state"]), 54)
        self.assertEqual(len(steps.loc[0, "end_state"]), 54)
        self.assertGreaterEqual(float(steps.loc[0, "action_probability"]), 0.0)
        self.assertLessEqual(float(steps.loc[0, "action_probability"]), 1.0)

    def test_one_hot_checkpoint_round_trips_encoding_config(self):
        model = ActorCriticNet(NetworkConfig(hidden_layers=(16,)))
        config = PPOConfig(n_epochs=1, n_steps=4, batch_size=2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model=model,
                config=config,
                network_config=model.config,
            )

            loaded_model, checkpoint = load_checkpoint(path)

        self.assertEqual(loaded_model.config.input_dim, ONE_HOT_OBSERVATION_SIZE)
        self.assertEqual(loaded_model.config.observation_encoding, "one_hot")
        self.assertEqual(checkpoint["checkpoint_version"], 2)
        self.assertEqual(checkpoint["network_config"]["observation_encoding"], "one_hot")

    def test_old_normalized_checkpoint_without_encoding_loads(self):
        model = ActorCriticNet(
            NetworkConfig(
                input_dim=OBSERVATION_SIZE,
                hidden_layers=(16,),
                observation_encoding="normalized",
            )
        )
        _make_actor_and_critic_hidden_layers_identical(model)
        old_network_config = {
            "input_dim": OBSERVATION_SIZE,
            "hidden_layers": [16],
            "actor_output_dim": 12,
            "critic_output_dim": 1,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_model.pt"
            torch.save(
                {
                    "model_state_dict": _legacy_shared_state_dict(model),
                    "ppo_config": {},
                    "network_config": old_network_config,
                    "metadata": {},
                },
                path,
            )

            loaded_model, _ = load_checkpoint(path)

        self.assertEqual(loaded_model.config.input_dim, OBSERVATION_SIZE)
        self.assertEqual(loaded_model.config.observation_encoding, "normalized")

    def test_legacy_shared_checkpoint_migration_preserves_predictions(self):
        torch.manual_seed(7)
        model = ActorCriticNet(NetworkConfig(hidden_layers=(16, 8)))
        _make_actor_and_critic_hidden_layers_identical(model)
        observations = torch.randn((4, ONE_HOT_OBSERVATION_SIZE))
        model.eval()
        with torch.no_grad():
            expected_logits, expected_values = model(observations)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_model.pt"
            torch.save(
                {
                    "model_state_dict": _legacy_shared_state_dict(model),
                    "ppo_config": {},
                    "network_config": {
                        **model.config.__dict__,
                        "hidden_layers": list(model.config.hidden_layers),
                    },
                    "metadata": {},
                },
                path,
            )

            loaded_model, checkpoint = load_checkpoint(path)
            with torch.no_grad():
                actual_logits, actual_values = loaded_model(observations)

        self.assertNotIn("checkpoint_version", checkpoint)
        self.assertTrue(torch.equal(actual_logits, expected_logits))
        self.assertTrue(torch.equal(actual_values, expected_values))


def _make_actor_and_critic_hidden_layers_identical(model: ActorCriticNet) -> None:
    output_layer_index = 2 * len(model.config.hidden_layers)
    with torch.no_grad():
        for layer_index in range(0, output_layer_index, 2):
            model.critic[layer_index].load_state_dict(
                model.actor[layer_index].state_dict()
            )


def _legacy_shared_state_dict(
    model: ActorCriticNet,
) -> dict[str, torch.Tensor]:
    output_layer_index = 2 * len(model.config.hidden_layers)
    legacy: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.startswith("actor."):
            suffix = key.removeprefix("actor.")
            layer_index, parameter_name = suffix.split(".", maxsplit=1)
            if int(layer_index) == output_layer_index:
                legacy[f"actor.{parameter_name}"] = value
            else:
                legacy[f"shared.{suffix}"] = value
        elif key.startswith(f"critic.{output_layer_index}."):
            parameter_name = key.removeprefix(
                f"critic.{output_layer_index}."
            )
            legacy[f"critic.{parameter_name}"] = value
    return legacy


def _row_for_moves(moves: str, depth: int | None = None) -> dict:
    cube = CubeState.solved()
    tokens = moves.split()
    for move in tokens:
        cube = apply_move(cube, move)
    solution = " ".join(_inverse_token(move) for move in reversed(tokens))
    first_move = solution.split()[0] if solution else ""
    return {
        "sample_id": f"sample_{moves.replace(' ', '_').replace(chr(39), 'prime')}",
        "scramble_depth": depth if depth is not None else len(tokens),
        "scramble_moves": moves,
        "state_encoded": cube.to_flat_string(),
        "solution_moves": solution,
        "first_solution_move": first_move,
    }


def _inverse_token(move: str) -> str:
    return move[:-1] if move.endswith("'") else f"{move}'"


def _write_depth_csv(directory: Path, depth: int, rows: list[dict]) -> Path:
    path = directory / f"depth_{depth}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    unittest.main()

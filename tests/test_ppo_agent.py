import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.ppo_agent import (  # noqa: E402
    ActorCriticNet,
    NetworkConfig,
    PPOConfig,
    RolloutBuffer,
    evaluate_ppo_model,
    evaluate_random_baseline,
    ppo_update,
    predict_action,
    preprocess_observations,
    select_action,
)
from cube.gym_environment import SOLVED_STATE_STRING, decode_state
from cube.moves import apply_move
from cube.state import CubeState


class AlwaysRPrimePolicy:
    def predict(self, observation, deterministic=True):
        return 3, None


class PPOAgentTests(unittest.TestCase):
    def test_network_forward_shapes(self):
        model = ActorCriticNet(NetworkConfig(hidden_layers=(32, 16)))
        observations = torch.zeros((4, 54), dtype=torch.float32)

        logits, values = model(observations)

        self.assertEqual(tuple(logits.shape), (4, 12))
        self.assertEqual(tuple(values.shape), (4,))

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

    def test_preprocess_normalizes_observations(self):
        observation = np.array([0, 5] * 27, dtype=np.int8)

        tensor = preprocess_observations(observation)

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

        metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            config=PPOConfig(n_epochs=2, batch_size=2),
            device=torch.device("cpu"),
        )

        changed = any(
            not torch.allclose(previous, current)
            for previous, current in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        self.assertIn("policy_loss", metrics)
        self.assertIn("explained_variance", metrics)

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

        self.assertEqual(results["by_depth"]["1"]["episodes"], 2)
        self.assertEqual(results["by_depth"]["1"]["solve_rate"], 1.0)
        self.assertEqual(results["overall"]["solved_count"], 2)

    def test_random_baseline_reports_expected_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_depth_csv(data_dir, 1, [_row_for_moves("R")])

            results = evaluate_random_baseline(
                depths=(1,),
                episodes_per_depth=1,
                max_episode_steps=2,
                data_dir=data_dir,
                seed=3,
            )

        self.assertIn("1", results["by_depth"])
        self.assertIn("solve_rate", results["overall"])
        self.assertIn("inverse_move_rate", results["overall"])


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
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    unittest.main()

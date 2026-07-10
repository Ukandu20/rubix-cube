from __future__ import annotations

import sys
import shutil
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cube.environment import ACTION_TO_MOVE, MOVE_TO_ACTION  # noqa: E402
from cube.moves import apply_move  # noqa: E402
from cube.state import CubeState  # noqa: E402
from showcase.service import (  # noqa: E402
    ActionDecision,
    RunConfig,
    benchmark_to_csv,
    discover_checkpoints,
    load_curriculum_weights,
    run_benchmark,
    run_solver,
    sample_scramble,
)
from showcase.visualization import cube_net_html  # noqa: E402


def decision(action: int) -> ActionDecision:
    probabilities = [0.0] * 12
    probabilities[action] = 1.0
    return ActionDecision(action=action, probabilities=tuple(probabilities))


class FixedPolicy:
    def __init__(self, greedy_action: int, stochastic_action: int | None = None):
        self.greedy_action = greedy_action
        self.stochastic_action = (
            greedy_action if stochastic_action is None else stochastic_action
        )

    def decide(self, state, *, greedy, temperature, rng):
        del state, temperature, rng
        return decision(self.greedy_action if greedy else self.stochastic_action)


class TickingPolicy(FixedPolicy):
    def __init__(self, clock):
        super().__init__(MOVE_TO_ACTION["R"])
        self.clock = clock

    def decide(self, state, *, greedy, temperature, rng):
        self.clock.value += 1.0
        return super().decide(
            state,
            greedy=greedy,
            temperature=temperature,
            rng=rng,
        )


class FakeClock:
    value = 0.0

    def __call__(self):
        return self.value


class ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = load_curriculum_weights(
            PROJECT_ROOT / "config" / "curriculum_config_depth_1_10.yaml"
        )

    def test_seeded_scramble_modes_are_reproducible(self):
        first = sample_scramble(
            mode="range", minimum=2, maximum=7, seed=123
        )
        second = sample_scramble(
            mode="range", minimum=2, maximum=7, seed=123
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 2)
        self.assertLessEqual(first[0], 7)

        exact = sample_scramble(
            mode="exact", minimum=6, maximum=6, seed=9
        )
        self.assertEqual(exact[0], 6)
        self.assertEqual(len(exact[1].split()), 6)

        curriculum = sample_scramble(
            mode="curriculum",
            minimum=8,
            maximum=10,
            seed=42,
            curriculum_weights=self.weights,
            curriculum_stage=10,
        )
        self.assertIn(curriculum[0], {8, 9, 10})

    def test_curriculum_weights_are_normalized(self):
        self.assertEqual(set(self.weights), set(range(1, 11)))
        for stage, weights in self.weights.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertTrue(all(depth <= stage for depth in weights))

    def test_greedy_solver_success(self):
        result = run_solver(
            FixedPolicy(MOVE_TO_ACTION["U'"]),
            scramble_length=1,
            scramble="U",
            config=RunConfig(max_moves=3, max_attempts=3, seed=4),
        )
        self.assertTrue(result.solved)
        self.assertTrue(result.greedy_solved)
        self.assertEqual(result.attempts_used, 1)
        self.assertEqual(result.display_attempt.moves, ("U'",))

    def test_stochastic_retry_uses_same_scramble(self):
        result = run_solver(
            FixedPolicy(MOVE_TO_ACTION["R"], MOVE_TO_ACTION["U'"]),
            scramble_length=1,
            scramble="U",
            config=RunConfig(max_moves=5, max_attempts=2, seed=5),
        )
        self.assertTrue(result.solved)
        self.assertFalse(result.greedy_solved)
        self.assertEqual(result.attempts_used, 2)
        self.assertEqual(result.attempts[0].termination_reason, "cycle")
        self.assertEqual(result.attempts[1].moves, ("U'",))
        self.assertEqual(
            result.attempts[0].frames[0].state,
            result.attempts[1].frames[0].state,
        )

    def test_move_limit_and_cycle_detection(self):
        limited = run_solver(
            FixedPolicy(MOVE_TO_ACTION["R"]),
            scramble_length=1,
            scramble="U",
            config=RunConfig(max_moves=2, max_attempts=1),
        )
        self.assertEqual(limited.attempts[0].termination_reason, "move_limit")
        self.assertEqual(limited.termination_reason, "attempt_limit")

        cycled = run_solver(
            FixedPolicy(MOVE_TO_ACTION["R"]),
            scramble_length=1,
            scramble="U",
            config=RunConfig(max_moves=8, max_attempts=1),
        )
        self.assertEqual(cycled.attempts[0].termination_reason, "cycle")

    def test_cooperative_timeout(self):
        clock = FakeClock()
        result = run_solver(
            TickingPolicy(clock),
            scramble_length=1,
            scramble="U",
            config=RunConfig(
                max_moves=10,
                max_attempts=3,
                timeout_seconds=0.5,
            ),
            clock=clock,
        )
        self.assertEqual(result.termination_reason, "time_limit")
        self.assertEqual(result.attempts[0].termination_reason, "time_limit")

    def test_cube_net_face_order_and_move_invariants(self):
        html = cube_net_html(CubeState.solved().to_flat_string())
        for face in ("U", "L", "F", "R", "B", "D"):
            self.assertIn(f">{face}</div>", html)

        solved = CubeState.solved()
        for move in ACTION_TO_MOVE.values():
            moved = apply_move(solved, move)
            inverse = move[:-1] if move.endswith("'") else f"{move}'"
            self.assertEqual(apply_move(moved, inverse), solved)
            repeated = solved
            for _ in range(4):
                repeated = apply_move(repeated, move)
            self.assertEqual(repeated, solved)

    def test_checkpoint_discovery_reports_validation_and_progress(self):
        root = (
            PROJECT_ROOT
            / "models"
            / "artifacts"
            / "ppo"
            / "depth_1_10_onehot"
        )
        if not root.exists():
            self.skipTest("local depth-1–10 artifact is not available")
        found = discover_checkpoints(root)
        self.assertGreaterEqual(len(found), 1)
        self.assertTrue(found[0].compatible)
        self.assertGreaterEqual(found[0].validated_depth, 1)
        self.assertGreaterEqual(
            found[0].run_curriculum_depth,
            found[0].checkpoint_curriculum_depth,
        )

    def test_checkpoint_discovery_prefers_more_validated_final_checkpoint(self):
        root = PROJECT_ROOT / ".tmp_showcase_checkpoint_ranking"
        artifact_dir = root / "depth_1_10_onehot" / "v999"
        if root.exists():
            shutil.rmtree(root)
        artifact_dir.mkdir(parents=True)
        try:
            base = {
                "network_config": {
                    "observation_encoding": "one_hot",
                    "actor_output_dim": 12,
                    "input_dim": 324,
                },
            }
            torch.save(
                {
                    **base,
                    "metadata": {
                        "timesteps": 10_240,
                        "evaluation": {"by_depth": {"1": {}}},
                        "curriculum": {"current_depth": 2},
                    },
                },
                artifact_dir / "best_model.pt",
            )
            torch.save(
                {
                    **base,
                    "metadata": {
                        "timesteps": 1_001_472,
                        "evaluation": {"by_depth": {"5": {}}},
                        "curriculum": {"current_depth": 5},
                    },
                },
                artifact_dir / "final_model.pt",
            )

            found = discover_checkpoints(root)
            self.assertEqual(
                found[0].label,
                "depth_1_10_onehot/v999/final_model.pt",
            )
            self.assertEqual(found[0].validated_depth, 5)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_benchmark_aggregation_and_csv(self):
        # U' solves a subset of single-turn scrambles; exact success is not
        # important here—the benchmark must aggregate every requested trial.
        benchmark = run_benchmark(
            FixedPolicy(MOVE_TO_ACTION["U'"]),
            lengths=[1, 2],
            trials_per_length=2,
            config=RunConfig(max_moves=2, max_attempts=1, seed=11),
        )
        self.assertEqual(len(benchmark.records), 4)
        self.assertEqual(len(benchmark.summary), 2)
        self.assertTrue(all(row["oracle_solved"] for row in benchmark.records))
        self.assertTrue(
            all(row["oracle_solve_rate"] == 1.0 for row in benchmark.summary)
        )
        self.assertIn("greedy_solve_rate", benchmark_to_csv(benchmark, summary=True))
        self.assertIn("oracle_moves", benchmark_to_csv(benchmark))


if __name__ == "__main__":
    unittest.main()

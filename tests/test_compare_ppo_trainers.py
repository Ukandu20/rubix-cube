import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.compare_ppo_trainers import (
    MATCHED_PPO_FIELDS,
    compare_runs,
    render_markdown,
)


class ComparePPOTrainersTests(unittest.TestCase):
    def _write_run(self, path: Path, *, trainer: str, gamma: float = 0.95) -> None:
        ppo = {field: 1 for field in MATCHED_PPO_FIELDS}
        ppo["gamma"] = gamma
        config = {
            "trainer": trainer,
            "seed": 42,
            "total_timesteps": 100,
            "actual_timesteps": 100,
            "ppo": ppo,
            "curriculum": {"min_depth": 1, "max_depth": 2},
            "training_session": {"elapsed_seconds": 10},
        }
        metrics = {
            "final_evaluation": {
                "overall": {
                    "solve_rate": 0.5,
                    "timeout_rate": 0.25,
                    "average_solution_length": 2,
                    "inverse_move_rate": 0.1,
                    "action_distribution": {"U": 0.5, "U'": 0.5},
                }
            },
            "curriculum_progress": {"current_depth": 2},
        }
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def test_matched_runs_are_comparable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root / "custom", trainer="custom")
            self._write_run(root / "sb3", trainer="stable_baselines3")
            result = compare_runs(root / "custom", root / "sb3")

        self.assertTrue(result["comparison_valid"])
        self.assertEqual(result["delta_sb3_minus_custom"]["solve_rate"], 0.0)
        self.assertIn("Configuration parity: **passed**", render_markdown(result))

    def test_hyperparameter_mismatch_invalidates_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root / "custom", trainer="custom", gamma=0.95)
            self._write_run(root / "sb3", trainer="stable_baselines3", gamma=0.99)
            result = compare_runs(root / "custom", root / "sb3")

        self.assertFalse(result["comparison_valid"])
        self.assertFalse(result["parity_checks"]["ppo.gamma"])


if __name__ == "__main__":
    unittest.main()

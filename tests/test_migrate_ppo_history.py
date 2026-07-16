import csv
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from agents.ppo_agent import write_history_csv  # noqa: E402
from scripts.migrate_ppo_history import migrate_metrics_file, migrate_tree  # noqa: E402


class PPOHistoryCsvTests(unittest.TestCase):
    def test_writer_preserves_columns_rows_and_nested_evaluation(self):
        history = [
            {"timesteps": 10, "loss": 0.5},
            {
                "timesteps": 20,
                "loss": 0.25,
                "evaluation": {"overall": {"solve_rate": 0.75}},
                "new_metric": 3,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_history_csv(path, history)

            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(
            reader.fieldnames,
            ["timesteps", "loss", "evaluation", "new_metric"],
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["timesteps"], "10")
        self.assertEqual(rows[0]["evaluation"], "")
        self.assertEqual(float(rows[1]["loss"]), 0.25)
        self.assertEqual(
            json.loads(rows[1]["evaluation"]),
            history[1]["evaluation"],
        )

    def test_migration_removes_only_history_and_is_idempotent(self):
        payload = {
            "history": [{"timesteps": 10, "loss": 0.5}],
            "best_solve_rate": 0.8,
            "final_evaluation": {"overall": {"solve_rate": 0.8}},
            "random_baseline": {"overall": {"solve_rate": 0.0}},
        }

        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "run" / "metrics.json"
            metrics_path.parent.mkdir()
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(migrate_metrics_file(metrics_path))
            self.assertFalse(migrate_metrics_file(metrics_path))

            migrated = json.loads(metrics_path.read_text(encoding="utf-8"))
            history_path = metrics_path.with_name("history.csv")
            history_exists = history_path.exists()

        self.assertNotIn("history", migrated)
        self.assertEqual(migrated["best_solve_rate"], 0.8)
        self.assertTrue(history_exists)

    def test_tree_migration_is_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ("first", "nested/second"):
                path = root / run / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps({"history": [{"timesteps": 1}]}),
                    encoding="utf-8",
                )
            skipped_path = root / "third" / "metrics.json"
            skipped_path.parent.mkdir()
            skipped_path.write_text(
                json.dumps({"best_solve_rate": 1.0}),
                encoding="utf-8",
            )

            migrated, skipped = migrate_tree(root)

            self.assertEqual((migrated, skipped), (2, 1))
            self.assertTrue((root / "first" / "history.csv").exists())
            self.assertTrue((root / "nested/second" / "history.csv").exists())

    def test_malformed_history_does_not_change_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            history_path = root / "history.csv"
            original_metrics = json.dumps({"history": "invalid"})
            metrics_path.write_text(original_metrics, encoding="utf-8")
            history_path.write_text("existing content", encoding="utf-8")

            with self.assertRaises(TypeError):
                migrate_metrics_file(metrics_path)

            self.assertEqual(
                metrics_path.read_text(encoding="utf-8"),
                original_metrics,
            )
            self.assertEqual(
                history_path.read_text(encoding="utf-8"),
                "existing content",
            )


if __name__ == "__main__":
    unittest.main()

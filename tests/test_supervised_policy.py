import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn

from cube.moves import apply_algorithm
from cube.state import CubeState
from data.training_data import FIELDNAMES, generate_training_example
from models.supervised_policy import (
    COLOR_ORDER,
    COLOR_TO_INDEX,
    INPUT_SIZE,
    LABEL_TO_INDEX,
    MOVE_ORDER,
    CubePolicyDataset,
    SupervisedPolicyNet,
    encode_state,
    evaluate_accuracy,
    greedy_solve,
    load_training_rows,
    save_artifacts,
    split_dataset,
    train_policy,
)
from scripts.convert_training_data_to_parquet import convert_training_data_to_parquet


class AlwaysRPrimeModel(nn.Module):
    def forward(self, inputs):
        logits = torch.zeros((inputs.shape[0], len(MOVE_ORDER)), dtype=torch.float32)
        logits[:, LABEL_TO_INDEX["R'"]] = 10.0
        return logits


class SupervisedPolicyTests(unittest.TestCase):
    def test_one_hot_encoder_shape_and_values(self):
        state = CubeState.solved().to_flat_string()
        encoded = encode_state(state)

        self.assertEqual(tuple(encoded.shape), (INPUT_SIZE,))
        self.assertEqual(float(encoded.sum().item()), 54.0)
        self.assertEqual(encoded[COLOR_TO_INDEX[state[0]]].item(), 1.0)
        self.assertEqual(
            encoded[len(COLOR_ORDER) + COLOR_TO_INDEX[state[1]]].item(),
            1.0,
        )

    def test_label_mapping_includes_all_legal_moves(self):
        self.assertEqual(
            MOVE_ORDER,
            ("U", "U'", "R", "R'", "F", "F'", "D", "D'", "L", "L'", "B", "B'"),
        )
        self.assertEqual(set(LABEL_TO_INDEX), set(MOVE_ORDER))
        self.assertEqual(len(LABEL_TO_INDEX), 12)

    def test_dataset_loader_reads_depth_csvs(self):
        rows = [
            generate_training_example("sample-1", 1).to_row(),
            generate_training_example("sample-2", 1).to_row(),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth_1.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            loaded_rows = load_training_rows(directory)

        self.assertEqual(len(loaded_rows), 2)
        self.assertEqual(loaded_rows[0]["sample_id"], "sample-1")

    def test_dataset_encodes_features_lazily(self):
        rows = [generate_training_example("sample-1", 1).to_row()]

        dataset = CubePolicyDataset(rows)
        features, label, depth = dataset[0]

        self.assertFalse(hasattr(dataset, "features"))
        self.assertEqual(tuple(features.shape), (INPUT_SIZE,))
        self.assertEqual(tuple(dataset.labels.shape), (1,))
        self.assertEqual(int(label.item()), LABEL_TO_INDEX[rows[0]["first_solution_move"]])
        self.assertEqual(int(depth.item()), 1)

    def test_loader_filters_depth_range(self):
        with tempfile.TemporaryDirectory() as directory:
            for depth in range(1, 5):
                rows = [generate_training_example(f"sample-{depth}", depth).to_row()]
                self._write_depth_csv(directory, depth, rows)

            loaded_rows = load_training_rows(directory, min_depth=1, max_depth=3)

        self.assertEqual({int(row["scramble_depth"]) for row in loaded_rows}, {1, 2, 3})

    def test_loader_applies_global_row_cap(self):
        rows = [
            generate_training_example(f"sample-{index}", 1).to_row()
            for index in range(10)
        ]

        with tempfile.TemporaryDirectory() as directory:
            self._write_depth_csv(directory, 1, rows)

            loaded_rows = load_training_rows(directory, max_rows=3, seed=12)

        self.assertEqual(len(loaded_rows), 3)

    def test_loader_applies_sample_per_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            for depth in (1, 2):
                rows = [
                    generate_training_example(f"sample-{depth}-{index}", depth).to_row()
                    for index in range(5)
                ]
                self._write_depth_csv(directory, depth, rows)

            loaded_rows = load_training_rows(directory, sample_per_depth=2, seed=12)

        counts = {1: 0, 2: 0}
        for row in loaded_rows:
            counts[int(row["scramble_depth"])] += 1
        self.assertEqual(counts, {1: 2, 2: 2})

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") is not None,
        "pyarrow is required for Parquet round-trip tests",
    )
    def test_loader_reads_converted_parquet_shards(self):
        rows = [
            generate_training_example("sample-1", 1).to_row(),
            generate_training_example("sample-2", 1).to_row(),
        ]

        with tempfile.TemporaryDirectory() as directory:
            self._write_depth_csv(directory, 1, rows)
            paths = convert_training_data_to_parquet(directory)

            loaded_rows = load_training_rows(directory, data_format="parquet")

        self.assertEqual([path.name for path in paths], ["depth_1.parquet"])
        self.assertEqual(len(loaded_rows), 2)
        self.assertEqual(loaded_rows[0]["sample_id"], "sample-1")
        self.assertEqual(loaded_rows[0]["first_solution_move"], rows[0]["first_solution_move"])

    def test_model_forward_shape(self):
        model = SupervisedPolicyNet()
        inputs = torch.zeros((4, INPUT_SIZE), dtype=torch.float32)

        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (4, 12))

    def test_training_loop_runs_one_tiny_epoch(self):
        rows = [generate_training_example(f"sample-{index}", 1).to_row() for index in range(6)]
        dataset = CubePolicyDataset(rows)
        train_dataset, validation_dataset = split_dataset(dataset, seed=1)
        model = SupervisedPolicyNet(hidden_size=32)

        history = train_policy(
            model,
            train_dataset,
            validation_dataset,
            epochs=1,
            batch_size=2,
        )

        self.assertEqual(len(history), 1)
        self.assertIn("train_loss", history[0])
        self.assertIn("validation_accuracy", history[0])

    def test_greedy_solver_solves_one_move_state_with_stub_model(self):
        cube = apply_algorithm(CubeState.solved(), "R")

        result = greedy_solve(AlwaysRPrimeModel(), cube.to_flat_string(), solve_cap=1)

        self.assertTrue(result["solved"])
        self.assertEqual(result["moves"], ["R'"])
        self.assertEqual(result["move_count"], 1)

    def test_metrics_include_overall_and_per_depth_accuracy(self):
        rows = [
            generate_training_example("sample-1", 1).to_row(),
            generate_training_example("sample-2", 2).to_row(),
        ]
        dataset = CubePolicyDataset(rows)
        model = SupervisedPolicyNet(hidden_size=32)

        metrics = evaluate_accuracy(model, dataset, batch_size=1)

        self.assertIn("accuracy", metrics)
        self.assertIn("accuracy_by_depth", metrics)
        self.assertEqual(set(metrics["accuracy_by_depth"]), {"1", "2"})

    def test_save_artifacts_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = save_artifacts(
                SupervisedPolicyNet(hidden_size=32),
                directory,
                config={"epochs": 1},
                metrics={"accuracy": 0.0},
            )

            self.assertTrue(paths["policy"].exists())
            self.assertTrue(paths["label_mapping"].exists())
            self.assertTrue(paths["config"].exists())
            self.assertTrue(paths["metrics"].exists())

    def _write_depth_csv(self, directory: str, depth: int, rows: list[dict]) -> Path:
        path = Path(directory) / f"depth_{depth}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        return path


if __name__ == "__main__":
    unittest.main()

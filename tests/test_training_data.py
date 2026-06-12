import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cube.moves import apply_algorithm
from cube.notation import split_algorithm
from cube.state import CubeState
from data.training_data import (
    FIELDNAMES,
    generate_depth_dataset,
    generate_training_datasets,
    generate_training_example,
    write_depth_csv,
)


class TrainingDataTests(unittest.TestCase):
    def test_generated_example_has_required_fields(self):
        example = generate_training_example("sample-1", depth=2, rng=random.Random(1))
        row = example.to_row()

        self.assertEqual(tuple(row), FIELDNAMES)
        self.assertEqual(example.sample_id, "sample-1")
        self.assertEqual(example.scramble_depth, 2)
        self.assertEqual(len(example.state_encoded), 54)
        self.assertEqual(example.first_solution_move, split_algorithm(example.solution_moves)[0])
        self.assertFalse(example.is_solved)

    def test_solution_moves_solve_generated_state(self):
        example = generate_training_example("sample-1", depth=3, rng=random.Random(2))
        cube = CubeState.from_flat_string(example.state_encoded, 3)
        solved = apply_algorithm(cube, example.solution_moves)

        self.assertTrue(solved.is_solved())

    def test_scramble_has_no_same_face_consecutive_moves(self):
        example = generate_training_example("sample-1", depth=5, rng=random.Random(3))
        moves = split_algorithm(example.scramble_moves)

        self.assertEqual(len(moves), 5)
        for current_move, next_move in zip(moves, moves[1:]):
            self.assertNotEqual(current_move[0], next_move[0])

    def test_unique_depth_dataset_enforces_unique_states(self):
        examples = generate_depth_dataset(
            depth=1,
            count=4,
            rng=random.Random(4),
            unique=True,
        )
        states = [example.state_encoded for example in examples]

        self.assertEqual(len(states), len(set(states)))

    def test_unique_depth_dataset_raises_when_retry_limit_is_exhausted(self):
        with self.assertRaisesRegex(ValueError, "unique examples"):
            generate_depth_dataset(
                depth=1,
                count=13,
                rng=random.Random(5),
                unique=True,
                max_attempts=50,
            )

    def test_write_depth_csv_creates_headers_and_rows(self):
        examples = generate_depth_dataset(
            depth=2,
            count=3,
            rng=random.Random(6),
            unique=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = write_depth_csv(examples, directory)

            self.assertEqual(output_path.name, "depth_2.csv")
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            self.assertEqual(tuple(rows[0]), FIELDNAMES)

    def test_generate_training_datasets_with_tiny_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_training_datasets(
                depth_counts={1: 2, 2: 2},
                output_dir=directory,
                rng=random.Random(7),
                unique=True,
            )

            self.assertEqual([path.name for path in paths], ["depth_1.csv", "depth_2.csv"])
            for path in paths:
                self.assertTrue(path.exists())

    def test_duplicate_generation_allows_large_shallow_counts(self):
        examples = generate_depth_dataset(
            depth=1,
            count=20,
            rng=random.Random(8),
            unique=False,
        )

        self.assertEqual(len(examples), 20)


if __name__ == "__main__":
    unittest.main()

import unittest

from cube.moves import apply_algorithm, apply_move, apply_moves
from cube.notation import Move, inverse_algorithm
from cube.state import CubeState

BASE_MOVES = ("U", "R", "F", "D", "L", "B")


class CubeMoveTests(unittest.TestCase):
    def test_base_moves_change_solved_cube_for_each_size(self):
        for move in BASE_MOVES:
            with self.subTest(move=move):
                cube = CubeState.solved()
                moved = apply_move(cube, move)

                self.assertNotEqual(moved, cube)
                self.assertFalse(moved.is_solved())

    def test_move_followed_by_inverse_restores_cube(self):
        for move in BASE_MOVES:
            with self.subTest(move=move):
                cube = CubeState.solved()
                restored = apply_algorithm(cube, f"{move} {move}'")

                self.assertEqual(restored, cube)

    def test_four_repeats_restore_cube(self):
        for move in BASE_MOVES:
            with self.subTest(move=move):
                cube = CubeState.solved()
                restored = apply_algorithm(cube, f"{move} {move} {move} {move}")

                self.assertEqual(restored, cube)

    def test_double_move_matches_two_single_moves(self):
        cube = CubeState.solved()

        self.assertEqual(apply_move(cube, "R2"), apply_algorithm(cube, "R R"))
        self.assertEqual(apply_algorithm(cube, "R2 U"), apply_algorithm(cube, "R R U"))

    def test_algorithm_and_inverse_restore_cube(self):
        algorithm = "R U R' U'"
        cube = CubeState.solved()
        moved = apply_algorithm(cube, algorithm)
        restored = apply_algorithm(moved, inverse_algorithm(algorithm))

        self.assertNotEqual(moved, cube)
        self.assertEqual(restored, cube)

    def test_move_functions_do_not_mutate_original_cube(self):
        cube = CubeState.solved()
        original_flat = cube.to_flat_string()

        moved = apply_algorithm(cube, "F R U")

        self.assertEqual(cube.to_flat_string(), original_flat)
        self.assertNotEqual(moved, cube)

    def test_apply_moves_accepts_parsed_moves(self):
        cube = CubeState.solved()

        self.assertEqual(
            apply_moves(cube, [Move("F"), Move("F", -1)]),
            cube,
        )

    def test_apply_moves_rejects_unparsed_tokens(self):
        with self.assertRaisesRegex(ValueError, "Move objects"):
            apply_moves(CubeState.solved(), ["R"])

    def test_wide_moves_are_rejected_by_executor(self):
        cube = CubeState.solved()

        with self.assertRaisesRegex(ValueError, "wide moves"):
            apply_move(cube, "Rw")
        with self.assertRaisesRegex(ValueError, "wide moves"):
            apply_algorithm(cube, "R Uw")


if __name__ == "__main__":
    unittest.main()

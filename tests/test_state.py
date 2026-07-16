import unittest
from collections import Counter

from cube.state import DEFAULT_COLORS, FACE_ORDER, CubeState


class CubeStateTests(unittest.TestCase):
    def test_solved_cube_shape_counts_and_status(self):
        cube = CubeState.solved()

        self.assertEqual(cube.size, 3)
        self.assertTrue(cube.is_solved())
        self.assertEqual(tuple(cube.faces), FACE_ORDER)
        self.assert_face_shapes(cube, 3)
        self.assert_sticker_counts(cube, 9)

    def test_non_three_by_three_sizes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "size"):
            CubeState.solved(2)
        with self.assertRaisesRegex(ValueError, "size"):
            CubeState.solved(4)

    def test_missing_face_is_rejected(self):
        faces = CubeState.solved().faces
        del faces["B"]

        with self.assertRaisesRegex(ValueError, "missing required cube face"):
            CubeState.from_faces(faces)

    def test_malformed_face_dimensions_are_rejected(self):
        faces = CubeState.solved().faces
        faces["U"][0].append("W")

        with self.assertRaisesRegex(ValueError, "exactly"):
            CubeState.from_faces(faces)

    def test_invalid_sticker_counts_are_rejected(self):
        faces = CubeState.solved().faces
        faces["U"][0][0] = "R"

        with self.assertRaisesRegex(ValueError, "must appear exactly 9 times"):
            CubeState.from_faces(faces)

    def test_flat_string_round_trip(self):
        cube = CubeState.solved()
        restored = CubeState.from_flat_string(cube.to_flat_string(), 3)

        self.assertEqual(restored, cube)
        self.assertTrue(restored.is_solved())

    def test_copy_equality_and_defensive_faces(self):
        cube = CubeState.solved(3)
        clone = cube.copy()

        self.assertEqual(clone, cube)
        self.assertIsNot(clone, cube)

        leaked_faces = cube.faces
        leaked_faces["U"][0][0] = "R"

        self.assertTrue(cube.is_solved())
        self.assertEqual(cube.faces["U"][0][0], DEFAULT_COLORS["U"])

    def assert_face_shapes(self, cube, size):
        for face in cube.faces.values():
            self.assertEqual(len(face), size)
            for row in face:
                self.assertEqual(len(row), size)

    def assert_sticker_counts(self, cube, expected_count):
        counts = Counter(
            sticker for face in cube.faces.values() for row in face for sticker in row
        )
        self.assertEqual(
            counts,
            Counter({color: expected_count for color in DEFAULT_COLORS.values()}),
        )


if __name__ == "__main__":
    unittest.main()

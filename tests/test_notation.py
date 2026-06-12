import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cube.notation import (
    Move,
    generate_scramble,
    inverse_algorithm,
    inverse_move,
    is_valid_move,
    normalize_algorithm,
    normalize_move,
    parse_algorithm,
    parse_move,
    split_algorithm,
)


class MoveNotationTests(unittest.TestCase):
    def test_parse_base_inverse_and_double_moves(self):
        self.assertEqual(parse_move("R"), [Move("R", 1, False)])
        self.assertEqual(parse_move("R'"), [Move("R", -1, False)])
        self.assertEqual(parse_move("R2"), [Move("R", 1, False), Move("R", 1, False)])

    def test_parse_wide_moves_and_expand_wide_double(self):
        self.assertEqual(parse_move("Rw"), [Move("R", 1, True)])
        self.assertEqual(parse_move("Rw'"), [Move("R", -1, True)])
        self.assertEqual(parse_move("Rw2"), [Move("R", 1, True), Move("R", 1, True)])

    def test_invalid_move_tokens_are_rejected(self):
        invalid_tokens = ("", "r", "u", "X", "R2'", "R'w", "RR", " R", "R ", "R  U")

        for token in invalid_tokens:
            with self.subTest(token=token):
                self.assertFalse(is_valid_move(token))
                with self.assertRaises(ValueError):
                    parse_move(token)

    def test_valid_move_names(self):
        valid_tokens = (
            "U",
            "R",
            "F",
            "D",
            "L",
            "B",
            "U'",
            "R2",
            "Uw",
            "Fw'",
            "Bw2",
        )

        for token in valid_tokens:
            with self.subTest(token=token):
                self.assertTrue(is_valid_move(token))

    def test_normalize_move_and_move_token(self):
        self.assertEqual(Move("F").to_token(), "F")
        self.assertEqual(Move("F", -1).to_token(), "F'")
        self.assertEqual(Move("F", 1, True).to_token(), "Fw")
        self.assertEqual(str(Move("F", -1, True)), "Fw'")
        self.assertEqual(normalize_move("R2"), "R R")
        self.assertEqual(normalize_move("Rw2"), "Rw Rw")

    def test_inverse_move(self):
        self.assertEqual(inverse_move(Move("F")), [Move("F", -1)])
        self.assertEqual(inverse_move("F"), [Move("F", -1)])
        self.assertEqual(inverse_move("F'"), [Move("F")])
        self.assertEqual(inverse_move("F2"), [Move("F", -1), Move("F", -1)])

    def test_split_parse_normalize_and_inverse_algorithm(self):
        self.assertEqual(split_algorithm("  R   U R'   U'  "), ["R", "U", "R'", "U'"])
        self.assertEqual(
            parse_algorithm("R U R' U'"),
            [Move("R"), Move("U"), Move("R", -1), Move("U", -1)],
        )
        self.assertEqual(parse_algorithm("R2 U"), [Move("R"), Move("R"), Move("U")])
        self.assertEqual(normalize_algorithm("R2   U"), "R R U")
        self.assertEqual(inverse_algorithm("R U R' U'"), "U R U' R'")
        self.assertEqual(inverse_algorithm("R2 U"), "U' R' R'")

    def test_empty_algorithm_is_valid_empty_sequence(self):
        self.assertEqual(split_algorithm("   "), [])
        self.assertEqual(parse_algorithm(""), [])
        self.assertEqual(normalize_algorithm("   "), "")
        self.assertEqual(inverse_algorithm(""), "")

    def test_generate_scramble_returns_empty_by_default(self):
        self.assertEqual(generate_scramble(), "")
        with self.assertRaisesRegex(ValueError, "size"):
            generate_scramble(size=2)

    def test_generate_scramble_with_explicit_length(self):
        scramble = generate_scramble(length=25, rng=random.Random(12))
        tokens = split_algorithm(scramble)

        self.assertEqual(len(parse_algorithm(scramble)), 25)
        self.assertEqual(len(tokens), 25)
        self.assertTrue(all(is_valid_move(token) for token in tokens))
        self.assertTrue(all("2" not in token for token in tokens))
        for current_token, next_token in zip(tokens, tokens[1:]):
            self.assertNotEqual(current_token[0], next_token[0])

    def test_invalid_scramble_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "size"):
            generate_scramble(size=4, length=1)
        with self.assertRaisesRegex(ValueError, "length"):
            generate_scramble(length=-1)


if __name__ == "__main__":
    unittest.main()

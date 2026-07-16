import unittest

from cube.encoding import decode_state, encode_state, validate_encoded_state
from cube.environment import ACTION_TO_MOVE, MOVE_TO_ACTION
from cube.episode import RewardConfig, advance_episode
from cube.state import CubeState


class EpisodeRulesTests(unittest.TestCase):
    def test_transition_applies_timeout_and_reward_in_one_place(self):
        transition = advance_episode(
            CubeState.solved(),
            MOVE_TO_ACTION["R"],
            action_to_move=ACTION_TO_MOVE,
            step_count=0,
            max_steps=1,
            reward_config=RewardConfig(timeout_penalty=-0.1),
        )

        self.assertFalse(transition.solved)
        self.assertTrue(transition.timed_out)
        self.assertEqual(transition.step_count, 1)
        self.assertAlmostEqual(transition.reward, -0.11)

    def test_transition_detects_an_immediate_inverse(self):
        transition = advance_episode(
            CubeState.solved(),
            MOVE_TO_ACTION["R'"],
            action_to_move=ACTION_TO_MOVE,
            step_count=1,
            max_steps=10,
            previous_action=MOVE_TO_ACTION["R"],
            inverse_action={MOVE_TO_ACTION["R"]: MOVE_TO_ACTION["R'"]},
            reward_config=RewardConfig(inverse_move_penalty=-0.05),
        )

        self.assertTrue(transition.immediate_inverse)
        self.assertAlmostEqual(transition.reward, -0.06)

    def test_observation_encoding_round_trip(self):
        encoded = CubeState.solved().to_flat_string()

        self.assertTrue(validate_encoded_state(encoded))
        self.assertEqual(encode_state(decode_state(encoded)), encoded)


if __name__ == "__main__":
    unittest.main()

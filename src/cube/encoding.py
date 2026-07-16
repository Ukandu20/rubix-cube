"""Conversion and validation for flat cube-state observations."""

from __future__ import annotations

from collections import Counter

import numpy as np

STICKER_COUNT = 54
COLOR_TO_INT = {"Y": 0, "O": 1, "G": 2, "W": 3, "R": 4, "B": 5}
INT_TO_COLOR = {value: key for key, value in COLOR_TO_INT.items()}


def decode_state(encoded_state: str) -> np.ndarray:
    """Decode a 54-character cube state string into integer sticker IDs."""

    encoded_state = encoded_state.strip()
    if not validate_encoded_state(encoded_state):
        raise ValueError("encoded_state must be a valid 54-sticker cube string")
    return np.array([COLOR_TO_INT[color] for color in encoded_state], dtype=np.int8)


def encode_state(decoded_state: np.ndarray) -> str:
    """Encode integer sticker IDs into a 54-character cube state string."""

    values = np.asarray(decoded_state).reshape(-1)
    if len(values) != STICKER_COUNT:
        raise ValueError(f"decoded_state must contain {STICKER_COUNT} stickers")
    invalid_values = set(int(value) for value in values) - set(INT_TO_COLOR)
    if invalid_values:
        raise ValueError(f"invalid sticker values: {invalid_values}")
    return "".join(INT_TO_COLOR[int(value)] for value in values)


def validate_encoded_state(encoded_state: str) -> bool:
    """Return whether an encoded state is structurally valid."""

    encoded_state = encoded_state.strip()
    if len(encoded_state) != STICKER_COUNT:
        return False
    if set(encoded_state) - set(COLOR_TO_INT):
        return False
    counts = Counter(encoded_state)
    return all(counts[color] == 9 for color in COLOR_TO_INT)

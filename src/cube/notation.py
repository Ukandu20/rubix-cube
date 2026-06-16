"""Move notation parsing and scramble utilities."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Optional, Union


BASE_FACES = ("U", "R", "F", "D", "L", "B")
_MOVE_PATTERN = re.compile(r"^([URFDLB])(w)?(2|')?$")
_SCRAMBLE_MOVES = tuple(f"{face}{suffix}" for face in BASE_FACES for suffix in ("", "'"))


@dataclass(frozen=True)
class Move:
    """One clockwise or counterclockwise quarter-turn action."""

    face: str
    direction: int = 1
    wide: bool = False

    def __post_init__(self) -> None:
        if self.face not in BASE_FACES:
            raise ValueError(f"unknown move face: {self.face!r}")
        if self.direction not in (1, -1):
            raise ValueError("move direction must be 1 or -1")
        if not isinstance(self.wide, bool):
            raise ValueError("move wide flag must be a boolean")

    def inverse(self) -> "Move":
        """Return the opposite quarter-turn."""

        return Move(self.face, -self.direction, self.wide)

    def to_token(self) -> str:
        """Return canonical notation for this single quarter-turn."""

        suffix = "'" if self.direction == -1 else ""
        wide = "w" if self.wide else ""
        return f"{self.face}{wide}{suffix}"

    def __str__(self) -> str:
        return self.to_token()


MoveInput = Union[str, Move]


def parse_move(token: str) -> list[Move]:
    """Parse one move token, expanding double-turn shorthand."""

    if not isinstance(token, str):
        raise ValueError("move token must be a string")
    if token == "":
        raise ValueError("move token cannot be empty")
    if token != token.strip():
        raise ValueError("move token cannot include surrounding whitespace")

    match = _MOVE_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"invalid move token: {token!r}")

    face, wide_marker, suffix = match.groups()
    wide = wide_marker == "w"
    direction = -1 if suffix == "'" else 1
    count = 2 if suffix == "2" else 1

    return [Move(face, direction, wide) for _ in range(count)]


def is_valid_move(token: str) -> bool:
    """Return whether a token is valid move notation."""

    try:
        parse_move(token)
    except ValueError:
        return False
    return True


def normalize_move(token: str) -> str:
    """Return canonical notation for one token, expanding double turns."""

    return " ".join(move.to_token() for move in parse_move(token))


def inverse_move(move: MoveInput) -> list[Move]:
    """Return the inverse move or expanded inverse moves for one token."""

    if isinstance(move, Move):
        return [move.inverse()]
    moves = parse_move(move)
    return [parsed_move.inverse() for parsed_move in reversed(moves)]


def split_algorithm(algorithm: str) -> list[str]:
    """Split an algorithm or scramble string into raw move tokens."""

    if not isinstance(algorithm, str):
        raise ValueError("algorithm must be a string")

    stripped = algorithm.strip()
    if not stripped:
        return []
    return stripped.split()


def parse_algorithm(algorithm: str) -> list[Move]:
    """Parse an algorithm string into individual quarter-turn moves."""

    moves: list[Move] = []
    for token in split_algorithm(algorithm):
        moves.extend(parse_move(token))
    return moves


def normalize_algorithm(algorithm: str) -> str:
    """Return canonical notation for an algorithm, expanding double turns."""

    return " ".join(move.to_token() for move in parse_algorithm(algorithm))


def inverse_algorithm(algorithm: str) -> str:
    """Return the inverse of an algorithm string."""

    moves = parse_algorithm(algorithm)
    return " ".join(move.inverse().to_token() for move in reversed(moves))


def generate_scramble(
    size: int = 3,
    length: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Generate an opt-in random face-turn scramble."""

    if size != 3:
        raise ValueError("scramble size must be 3")
    if length is None:
        return ""
    if length < 0:
        raise ValueError("scramble length cannot be negative")

    random_source = rng if rng is not None else random
    tokens: list[str] = []
    previous_token: Optional[str] = None

    for _ in range(length):
        if previous_token is None:
            choices = list(_SCRAMBLE_MOVES)
        else:
            blocked_token = inverse_move(previous_token)[0].to_token()
            choices = [move for move in _SCRAMBLE_MOVES if move != blocked_token]

        token = random_source.choice(choices)
        tokens.append(token)
        previous_token = token

    return " ".join(tokens)

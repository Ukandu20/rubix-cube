"""Sticker-based cube state for 3x3 cubes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

FaceName = str
Sticker = str
Face = tuple[tuple[Sticker, ...], ...]
Faces = dict[FaceName, Face]

FACE_ORDER: tuple[FaceName, ...] = ("U", "R", "F", "D", "L", "B")
DEFAULT_COLORS: Mapping[FaceName, Sticker] = {
    "U": "Y",
    "R": "O",
    "F": "G",
    "D": "W",
    "L": "R",
    "B": "B",
}
CUBE_SIZE = 3


class CubeState:
    """State for a 3x3 cube using six sticker faces."""

    __slots__ = ("_size", "_faces")

    def __init__(
        self,
        size: int = 3,
        faces: Mapping[FaceName, Sequence[Sequence[Sticker]]] | None = None,
    ) -> None:
        self._validate_size(size)
        normalized_faces = (
            self._solved_faces(size) if faces is None else self._normalize_faces(faces)
        )
        self._validate_faces(normalized_faces, size)
        self._size = size
        self._faces = normalized_faces

    @classmethod
    def solved(cls, size: int = CUBE_SIZE) -> CubeState:
        """Create a solved 3x3 cube."""

        return cls(size=size)

    @classmethod
    def from_faces(
        cls, faces: Mapping[FaceName, Sequence[Sequence[Sticker]]]
    ) -> CubeState:
        """Create a cube by inferring size from the supplied face grids."""

        size = cls._infer_size(faces)
        return cls(size=size, faces=faces)

    @classmethod
    def from_flat_string(cls, value: str, size: int) -> CubeState:
        """Create a cube from stickers ordered by U, R, F, D, L, B faces."""

        cls._validate_size(size)
        if not isinstance(value, str):
            raise ValueError("flat cube state must be a string")

        expected_length = len(FACE_ORDER) * size * size
        if len(value) != expected_length:
            raise ValueError(
                f"flat cube state must contain {expected_length} stickers "
                f"for a {size}x{size} cube"
            )

        faces: dict[FaceName, tuple[tuple[Sticker, ...], ...]] = {}
        cursor = 0
        for face_name in FACE_ORDER:
            rows = []
            for _ in range(size):
                row = tuple(value[cursor : cursor + size])
                rows.append(row)
                cursor += size
            faces[face_name] = tuple(rows)

        return cls(size=size, faces=faces)

    @property
    def size(self) -> int:
        return self._size

    @property
    def faces(self) -> dict[FaceName, list[list[Sticker]]]:
        """Return a defensive copy of the cube faces."""

        return {
            face_name: [list(row) for row in self._faces[face_name]]
            for face_name in FACE_ORDER
        }

    def validate(self) -> None:
        """Raise ValueError if the current state is not a valid sticker state."""

        self._validate_faces(self._faces, self._size)

    def is_solved(self) -> bool:
        """Return whether each face contains only its solved color."""

        return all(
            all(
                sticker == DEFAULT_COLORS[face_name]
                for row in self._faces[face_name]
                for sticker in row
            )
            for face_name in FACE_ORDER
        )

    def copy(self) -> CubeState:
        """Return an equal, independent cube state."""

        return CubeState(size=self._size, faces=self._faces)

    def to_flat_string(self) -> str:
        """Serialize stickers in U, R, F, D, L, B face order."""

        return "".join(
            sticker
            for face_name in FACE_ORDER
            for row in self._faces[face_name]
            for sticker in row
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubeState):
            return NotImplemented
        return self._size == other._size and self._faces == other._faces

    def __repr__(self) -> str:
        return f"CubeState(size={self._size}, flat={self.to_flat_string()!r})"

    @classmethod
    def _solved_faces(cls, size: int) -> Faces:
        return {
            face_name: tuple(
                tuple(DEFAULT_COLORS[face_name] for _ in range(size))
                for _ in range(size)
            )
            for face_name in FACE_ORDER
        }

    @classmethod
    def _normalize_faces(
        cls, faces: Mapping[FaceName, Sequence[Sequence[Sticker]]]
    ) -> Faces:
        if not isinstance(faces, Mapping):
            raise ValueError("faces must be a mapping of face names to sticker grids")

        missing = [face_name for face_name in FACE_ORDER if face_name not in faces]
        if missing:
            raise ValueError(f"missing required cube face(s): {', '.join(missing)}")

        extra = [face_name for face_name in faces if face_name not in FACE_ORDER]
        if extra:
            raise ValueError(f"unknown cube face(s): {', '.join(extra)}")

        return {
            face_name: tuple(
                tuple(sticker for sticker in row) for row in faces[face_name]
            )
            for face_name in FACE_ORDER
        }

    @classmethod
    def _infer_size(cls, faces: Mapping[FaceName, Sequence[Sequence[Sticker]]]) -> int:
        normalized_faces = cls._normalize_faces(faces)
        size = len(normalized_faces["U"])
        cls._validate_size(size)
        cls._validate_faces(normalized_faces, size)
        return size

    @staticmethod
    def _validate_size(size: int) -> None:
        if size != CUBE_SIZE:
            raise ValueError("cube size must be 3")

    @classmethod
    def _validate_faces(cls, faces: Mapping[FaceName, Face], size: int) -> None:
        missing = [face_name for face_name in FACE_ORDER if face_name not in faces]
        if missing:
            raise ValueError(f"missing required cube face(s): {', '.join(missing)}")

        for face_name in FACE_ORDER:
            face = faces[face_name]
            if len(face) != size:
                raise ValueError(f"face {face_name} must contain exactly {size} rows")
            for row in face:
                if len(row) != size:
                    raise ValueError(
                        f"face {face_name} rows must contain exactly {size} stickers"
                    )
                for sticker in row:
                    if sticker not in DEFAULT_COLORS.values():
                        raise ValueError(f"unknown sticker color: {sticker!r}")

        stickers = [
            sticker
            for face_name in FACE_ORDER
            for row in faces[face_name]
            for sticker in row
        ]
        expected_count = size * size
        counts = Counter(stickers)
        for color in DEFAULT_COLORS.values():
            if counts[color] != expected_count:
                raise ValueError(
                    f"color {color!r} must appear exactly {expected_count} times"
                )

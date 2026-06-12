"""Cube move execution for sticker-based cube states."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

from cube.notation import Move, parse_algorithm, parse_move
from cube.state import CubeState, FACE_ORDER, Sticker


Coordinate = Tuple[int, int, int]
Normal = Tuple[int, int, int]
MoveInput = Union[str, Move]


_MOVE_LAYERS = {
    "U": ("y", 1),
    "D": ("y", -1),
    "R": ("x", 1),
    "L": ("x", -1),
    "F": ("z", 1),
    "B": ("z", -1),
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def apply_move(cube: CubeState, move: MoveInput) -> CubeState:
    """Apply one move token or one parsed Move to a cube."""

    if isinstance(move, Move):
        return _apply_quarter_turn(cube, move)

    result = cube
    for parsed_move in parse_move(move):
        result = _apply_quarter_turn(result, parsed_move)
    return result


def apply_moves(cube: CubeState, moves: Iterable[Move]) -> CubeState:
    """Apply parsed quarter-turn moves in order."""

    result = cube
    for move in moves:
        if not isinstance(move, Move):
            raise ValueError("apply_moves expects Move objects")
        result = _apply_quarter_turn(result, move)
    return result


def apply_algorithm(cube: CubeState, algorithm: str) -> CubeState:
    """Apply a notation algorithm string to a cube."""

    return apply_moves(cube, parse_algorithm(algorithm))


def _apply_quarter_turn(cube: CubeState, move: Move) -> CubeState:
    if move.wide:
        raise ValueError("wide moves are not supported by the move executor yet")

    size = cube.size
    source_faces = cube.faces
    target_faces: dict[str, list[list[Optional[Sticker]]]] = {
        face_name: [[None for _ in range(size)] for _ in range(size)]
        for face_name in FACE_ORDER
    }
    axis, layer_sign = _MOVE_LAYERS[move.face]
    axis_index = _AXIS_INDEX[axis]
    layer_value = size - 1 if layer_sign == 1 else 0
    turns = (-layer_sign * move.direction) % 4

    for face_name in FACE_ORDER:
        for row_index, row in enumerate(source_faces[face_name]):
            for column_index, sticker in enumerate(row):
                coordinate, normal = _face_to_coordinate(
                    face_name, row_index, column_index, size
                )
                if coordinate[axis_index] == layer_value:
                    coordinate, normal = _rotate(coordinate, normal, axis, turns, size)
                target_face, target_row, target_column = _coordinate_to_face(
                    coordinate, normal, size
                )
                target_faces[target_face][target_row][target_column] = sticker

    return CubeState(
        size=size,
        faces={
            face_name: [
                [_require_sticker(sticker) for sticker in row]
                for row in target_faces[face_name]
            ]
            for face_name in FACE_ORDER
        },
    )


def _face_to_coordinate(
    face_name: str, row: int, column: int, size: int
) -> tuple[Coordinate, Normal]:
    last = size - 1
    if face_name == "F":
        return (column, last - row, last), (0, 0, 1)
    if face_name == "B":
        return (last - column, last - row, 0), (0, 0, -1)
    if face_name == "U":
        return (column, last, row), (0, 1, 0)
    if face_name == "D":
        return (column, 0, last - row), (0, -1, 0)
    if face_name == "R":
        return (last, last - row, last - column), (1, 0, 0)
    if face_name == "L":
        return (0, last - row, column), (-1, 0, 0)
    raise ValueError(f"unknown face: {face_name!r}")


def _coordinate_to_face(
    coordinate: Coordinate, normal: Normal, size: int
) -> tuple[str, int, int]:
    x, y, z = coordinate
    last = size - 1
    if normal == (0, 0, 1):
        return "F", last - y, x
    if normal == (0, 0, -1):
        return "B", last - y, last - x
    if normal == (0, 1, 0):
        return "U", z, x
    if normal == (0, -1, 0):
        return "D", last - z, x
    if normal == (1, 0, 0):
        return "R", last - y, last - z
    if normal == (-1, 0, 0):
        return "L", last - y, z
    raise ValueError(f"invalid sticker normal: {normal!r}")


def _rotate(
    coordinate: Coordinate, normal: Normal, axis: str, turns: int, size: int
) -> tuple[Coordinate, Normal]:
    rotated_coordinate = coordinate
    rotated_normal = normal
    for _ in range(turns):
        rotated_coordinate = _rotate_coordinate_positive(rotated_coordinate, axis, size)
        rotated_normal = _rotate_normal_positive(rotated_normal, axis)
    return rotated_coordinate, rotated_normal


def _rotate_coordinate_positive(
    coordinate: Coordinate, axis: str, size: int
) -> Coordinate:
    x, y, z = coordinate
    last = size - 1
    if axis == "x":
        return x, last - z, y
    if axis == "y":
        return z, y, last - x
    if axis == "z":
        return last - y, x, z
    raise ValueError(f"invalid rotation axis: {axis!r}")


def _rotate_normal_positive(normal: Normal, axis: str) -> Normal:
    x, y, z = normal
    if axis == "x":
        return x, -z, y
    if axis == "y":
        return z, y, -x
    if axis == "z":
        return -y, x, z
    raise ValueError(f"invalid rotation axis: {axis!r}")


def _require_sticker(sticker: Optional[Sticker]) -> Sticker:
    if sticker is None:
        raise ValueError("move produced an incomplete cube state")
    return sticker

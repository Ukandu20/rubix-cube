"""HTML rendering for a 3x3 cube net."""

from __future__ import annotations

from html import escape

from cube.state import CubeState

STICKER_COLORS = {
    "Y": "#facc15",
    "O": "#f97316",
    "G": "#22c55e",
    "W": "#f8fafc",
    "R": "#ef4444",
    "B": "#3b82f6",
}


def cube_net_html(state: str) -> str:
    """Render U / L F R B / D as a labelled, accessible HTML cube net."""

    cube = CubeState.from_flat_string(state, 3)
    faces = cube.faces
    positions = {
        "U": (1, 2),
        "L": (2, 1),
        "F": (2, 2),
        "R": (2, 3),
        "B": (2, 4),
        "D": (3, 2),
    }
    face_html: list[str] = []
    for face in ("U", "L", "F", "R", "B", "D"):
        row, column = positions[face]
        stickers = "".join(
            (
                '<span class="cube-sticker" '
                f'style="background:{STICKER_COLORS[sticker]}" '
                f'title="{escape(face)} {escape(sticker)}"></span>'
            )
            for face_row in faces[face]
            for sticker in face_row
        )
        face_html.append(
            f'<div class="cube-face" style="grid-row:{row};grid-column:{column}">'
            f'<div class="cube-face-label">{face}</div>'
            f'<div class="cube-face-grid">{stickers}</div></div>'
        )
    return (
        "<style>"
        ".cube-net{display:grid;grid-template-columns:repeat(4,108px);"
        "grid-template-rows:repeat(3,124px);gap:8px;justify-content:center;"
        "margin:0.5rem auto 1rem}.cube-face{text-align:center}"
        ".cube-face-label{font:600 12px sans-serif;margin-bottom:4px;color:#64748b}"
        ".cube-face-grid{display:grid;grid-template-columns:repeat(3,34px);"
        "grid-template-rows:repeat(3,34px);gap:2px;background:#111827;"
        "padding:3px;border-radius:5px;box-shadow:0 2px 8px #0003}"
        ".cube-sticker{display:block;border-radius:3px;border:1px solid #0004}"
        "@media(max-width:650px){.cube-net{transform:scale(.75);"
        "transform-origin:top center;margin-bottom:-80px}}"
        '</style><div class="cube-net">' + "".join(face_html) + "</div>"
    )

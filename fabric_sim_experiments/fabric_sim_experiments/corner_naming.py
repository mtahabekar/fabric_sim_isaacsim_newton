"""Corner names, labels and colors shared by the recorder and the plotters.

/fabric/sim/corners PoseArray order: 0 follower_left, 1 follower_right,
2 leader_left, 3 leader_right.
"""
from __future__ import annotations

CORNER_KEYS = ('follower_left', 'follower_right', 'leader_left', 'leader_right')
CORNER_SHORT_LABELS = {
    'follower_left': 'Follower L',
    'follower_right': 'Follower R',
    'leader_left': 'Leader L',
    'leader_right': 'Leader R',
}
CORNER_COLORS = {
    'follower_left': '#e41a1c',
    'follower_right': '#377eb8',
    'leader_left': '#4daf4a',
    'leader_right': '#ff7f00',
}


def corner_columns() -> list[str]:
    return ['t', *(f'{corner}_{axis}' for corner in CORNER_KEYS for axis in 'xyz')]


def row_xyz(row: dict[str, str], corner: str) -> tuple[float, float, float]:
    return tuple(float(row[f'{corner}_{axis}']) for axis in 'xyz')

"""Canonical leader/follower corner names used by experiment outputs.

The ROS topics still use the historical top/bottom names for compatibility.
CSV files and plot labels use these role names:

PoseArray index 0: follower_left   (legacy: bottom_left / BL)
PoseArray index 1: follower_right  (legacy: bottom_right / BR)
PoseArray index 2: leader_left     (legacy: top_left / TL)
PoseArray index 3: leader_right    (legacy: top_right / TR)
"""

from __future__ import annotations


CORNER_KEYS = ("follower_left", "follower_right", "leader_left", "leader_right")
LEGACY_KEYS = ("BL", "BR", "TL", "TR")
CORNER_LABELS = {
    "follower_left": "Follower left",
    "follower_right": "Follower right",
    "leader_left": "Leader left",
    "leader_right": "Leader right",
}
CORNER_SHORT_LABELS = {
    "follower_left": "Follower L",
    "follower_right": "Follower R",
    "leader_left": "Leader L",
    "leader_right": "Leader R",
}
CORNER_COLORS = {
    "follower_left": "#e41a1c",
    "follower_right": "#377eb8",
    "leader_left": "#4daf4a",
    "leader_right": "#ff7f00",
}
LEGACY_TO_CANONICAL = dict(zip(LEGACY_KEYS, CORNER_KEYS))
CANONICAL_TO_LEGACY = dict(zip(CORNER_KEYS, LEGACY_KEYS))


def corner_columns() -> list[str]:
    cols = ["t"]
    for corner in CORNER_KEYS:
        cols.extend([f"{corner}_x", f"{corner}_y", f"{corner}_z"])
    return cols


def row_xyz(row: dict[str, str], corner: str) -> tuple[float, float, float]:
    """Read a corner position from either canonical or legacy CSV headers."""
    legacy = CANONICAL_TO_LEGACY[corner]
    try:
        return (
            float(row[f"{corner}_x"]),
            float(row[f"{corner}_y"]),
            float(row[f"{corner}_z"]),
        )
    except KeyError:
        return (
            float(row[f"{legacy}_x"]),
            float(row[f"{legacy}_y"]),
            float(row[f"{legacy}_z"]),
        )

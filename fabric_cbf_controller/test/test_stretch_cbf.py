#!/usr/bin/env python3
"""Unit tests for the stretch-avoidance CBF math.

Tests are pure-numpy and require no ROS installation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pure-math helpers that mirror the logic in _build_stretch_constraints
# ---------------------------------------------------------------------------

def stretch_h(pos_i: np.ndarray, pos_j: np.ndarray, d_max: float) -> float:
    """Barrier value: positive = safe, negative = overstretched."""
    return d_max - float(np.linalg.norm(pos_j - pos_i))


def stretch_constraint(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    d_max: float,
    alpha_gain: float,
    v_j: np.ndarray,
    i_is_bl: bool,
) -> tuple[np.ndarray, float] | tuple[None, None]:
    """Return (row_6, rhs) for the CBF constraint row_6 @ u >= rhs."""
    diff = pos_j - pos_i
    dist = float(np.linalg.norm(diff))
    if dist < 1.0e-6:
        return None, None
    a_ij = diff / dist
    h = d_max - dist
    alpha_h = alpha_gain * h
    rhs = float(-alpha_h + np.dot(a_ij, v_j))
    row = (
        np.concatenate([a_ij, np.zeros(3)])
        if i_is_bl
        else np.concatenate([np.zeros(3), a_ij])
    )
    return row, rhs


# ---------------------------------------------------------------------------
# (a) h > 0 when dist < d_max  (safe)
# ---------------------------------------------------------------------------

def test_h_positive_when_safe():
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 0.7])   # dist = 0.7, d_max = 0.85
    h = stretch_h(follower_left, leader_left, d_max=0.85)
    assert h > 0.0
    assert math.isclose(h, 0.85 - 0.7, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# (b) h = 0 when dist = d_max  (boundary)
# ---------------------------------------------------------------------------

def test_h_zero_at_boundary():
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 0.85])  # dist exactly d_max
    h = stretch_h(follower_left, leader_left, d_max=0.85)
    assert math.isclose(h, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# (c) h < 0 when dist > d_max  (violated)
# ---------------------------------------------------------------------------

def test_h_negative_when_violated():
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 1.0])   # dist = 1.0 > d_max = 0.85
    h = stretch_h(follower_left, leader_left, d_max=0.85)
    assert h < 0.0
    assert math.isclose(h, 0.85 - 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# (d) Constraint row direction: a_ij points from i to j
# ---------------------------------------------------------------------------

def test_row_direction_bl_constraint():
    """For follower-left/leader-left, a_ij points from follower to leader."""
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 0.7])
    v_leader_left = np.zeros(3)
    row, _ = stretch_constraint(
        follower_left, leader_left, 0.85, 1.0, v_leader_left, i_is_bl=True
    )
    assert row is not None
    a_ij = row[0:3]
    expected = np.array([0.0, 0.0, 1.0])   # unit vector upward
    np.testing.assert_allclose(a_ij, expected, atol=1e-12)
    # follower-right half must be zero
    np.testing.assert_allclose(row[3:6], np.zeros(3), atol=1e-12)


def test_row_direction_br_constraint():
    """For follower-right/leader-right, a_ij points from follower to leader."""
    follower_right = np.array([0.0, 0.0, 0.0])
    leader_right = np.array([0.0, 0.0, 0.7])
    v_leader_right = np.zeros(3)
    row, _ = stretch_constraint(
        follower_right, leader_right, 0.85, 1.0, v_leader_right, i_is_bl=False
    )
    assert row is not None
    a_ij = row[3:6]
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(a_ij, expected, atol=1e-12)
    # follower-left half must be zero
    np.testing.assert_allclose(row[0:3], np.zeros(3), atol=1e-12)


def test_bl_br_rows_are_opposite():
    """Follower-left/follower-right pair produces opposite a_ij directions."""
    follower_left = np.array([-0.4, 0.0, 0.0])
    follower_right = np.array([0.4, 0.0, 0.0])
    v_follower_left = np.zeros(3)
    v_follower_right = np.zeros(3)
    row_follower_left, _ = stretch_constraint(
        follower_left, follower_right, 0.85, 1.0, v_follower_right, i_is_bl=True
    )
    row_follower_right, _ = stretch_constraint(
        follower_right, follower_left, 0.85, 1.0, v_follower_left, i_is_bl=False
    )
    assert row_follower_left is not None and row_follower_right is not None
    # follower-left to follower-right is +x; reverse direction is -x.
    np.testing.assert_allclose(row_follower_left[0:3], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(row_follower_right[3:6], [-1.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# (e) Velocity compensation term sign
# ---------------------------------------------------------------------------

def test_velocity_compensation_sign():
    """Positive v_j projected onto a_ij increases rhs (tightens constraint)."""
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 0.7])   # a_ij = [0,0,1], h = 0.15 > 0
    alpha_gain = 1.0
    v_leader_left_zero = np.zeros(3)
    v_leader_left_up = np.array([0.0, 0.0, 0.2])   # moving away from follower-left

    _, rhs_no_vel = stretch_constraint(
        follower_left, leader_left, 0.85, alpha_gain, v_leader_left_zero, True
    )
    _, rhs_with_vel = stretch_constraint(
        follower_left, leader_left, 0.85, alpha_gain, v_leader_left_up, True
    )

    # If leader-left moves away, follower-left must keep up, so rhs is larger.
    assert rhs_with_vel > rhs_no_vel
    assert math.isclose(rhs_with_vel - rhs_no_vel, 0.2, rel_tol=1e-9)


def test_velocity_compensation_opposite_direction():
    """Negative v_j (j approaching i) relaxes the constraint."""
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 0.7])
    v_leader_left_down = np.array([0.0, 0.0, -0.2])   # moving toward follower-left

    _, rhs_no_vel = stretch_constraint(follower_left, leader_left, 0.85, 1.0, np.zeros(3), True)
    _, rhs_closing = stretch_constraint(
        follower_left, leader_left, 0.85, 1.0, v_leader_left_down, True
    )

    assert rhs_closing < rhs_no_vel


# ---------------------------------------------------------------------------
# Additional: rhs forces convergence when violated
# ---------------------------------------------------------------------------

def test_rhs_positive_when_violated():
    """When h < 0 (fabric overstretched), rhs > 0 forces u toward j."""
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 1.0])   # dist=1.0 > d_max=0.85 -> h=-0.15
    _, rhs = stretch_constraint(follower_left, leader_left, 0.85, 1.0, np.zeros(3), True)
    # rhs = -alpha*h = -1*(-0.15) = 0.15 > 0
    assert rhs > 0.0
    assert math.isclose(rhs, 0.15, rel_tol=1e-9)


def test_rhs_negative_when_safe():
    """When h > 0 (fabric within limit), rhs < 0 (constraint easily satisfied)."""
    follower_left = np.array([0.0, 0.0, 0.0])
    leader_left = np.array([0.0, 0.0, 0.7])   # dist=0.7, h=0.15
    _, rhs = stretch_constraint(follower_left, leader_left, 0.85, 1.0, np.zeros(3), True)
    # rhs = -1*0.15 = -0.15 < 0
    assert rhs < 0.0
    assert math.isclose(rhs, -0.15, rel_tol=1e-9)


def test_degenerate_zero_distance_returns_none():
    """Coincident positions should return None (skip constraint)."""
    pos = np.array([0.5, 0.5, 0.5])
    row, rhs = stretch_constraint(pos, pos.copy(), 0.85, 1.0, np.zeros(3), True)
    assert row is None and rhs is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

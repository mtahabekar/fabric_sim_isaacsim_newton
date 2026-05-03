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
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 0.7])   # dist = 0.7, d_max = 0.85
    h = stretch_h(bl, tl, d_max=0.85)
    assert h > 0.0
    assert math.isclose(h, 0.85 - 0.7, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# (b) h = 0 when dist = d_max  (boundary)
# ---------------------------------------------------------------------------

def test_h_zero_at_boundary():
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 0.85])  # dist exactly d_max
    h = stretch_h(bl, tl, d_max=0.85)
    assert math.isclose(h, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# (c) h < 0 when dist > d_max  (violated)
# ---------------------------------------------------------------------------

def test_h_negative_when_violated():
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 1.0])   # dist = 1.0 > d_max = 0.85
    h = stretch_h(bl, tl, d_max=0.85)
    assert h < 0.0
    assert math.isclose(h, 0.85 - 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# (d) Constraint row direction: a_ij points from i to j
# ---------------------------------------------------------------------------

def test_row_direction_bl_constraint():
    """For a BL-TL pair, a_ij should point from BL toward TL (upward)."""
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 0.7])
    v_tl = np.zeros(3)
    row, _ = stretch_constraint(bl, tl, 0.85, 1.0, v_tl, i_is_bl=True)
    assert row is not None
    a_ij = row[0:3]
    expected = np.array([0.0, 0.0, 1.0])   # unit vector upward
    np.testing.assert_allclose(a_ij, expected, atol=1e-12)
    # BR half must be zero
    np.testing.assert_allclose(row[3:6], np.zeros(3), atol=1e-12)


def test_row_direction_br_constraint():
    """For a BR-TR pair, a_ij should point from BR toward TR; row in slot 3:6."""
    br = np.array([0.0, 0.0, 0.0])
    tr = np.array([0.0, 0.0, 0.7])
    v_tr = np.zeros(3)
    row, _ = stretch_constraint(br, tr, 0.85, 1.0, v_tr, i_is_bl=False)
    assert row is not None
    a_ij = row[3:6]
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(a_ij, expected, atol=1e-12)
    # BL half must be zero
    np.testing.assert_allclose(row[0:3], np.zeros(3), atol=1e-12)


def test_bl_br_rows_are_opposite():
    """BL-BR pair produces two rows with opposite a_ij directions."""
    bl = np.array([-0.4, 0.0, 0.0])
    br = np.array([0.4, 0.0, 0.0])
    v_bl = np.zeros(3)
    v_br = np.zeros(3)
    row_bl, _ = stretch_constraint(bl, br, 0.85, 1.0, v_br, i_is_bl=True)
    row_br, _ = stretch_constraint(br, bl, 0.85, 1.0, v_bl, i_is_bl=False)
    assert row_bl is not None and row_br is not None
    # a_BL_BR = +x, a_BR_BL = -x
    np.testing.assert_allclose(row_bl[0:3], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(row_br[3:6], [-1.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# (e) Velocity compensation term sign
# ---------------------------------------------------------------------------

def test_velocity_compensation_sign():
    """Positive v_j projected onto a_ij increases rhs (tightens constraint)."""
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 0.7])   # a_ij = [0,0,1], h = 0.15 > 0
    alpha_gain = 1.0
    v_tl_zero = np.zeros(3)
    v_tl_up = np.array([0.0, 0.0, 0.2])   # TL moving away from BL

    _, rhs_no_vel = stretch_constraint(bl, tl, 0.85, alpha_gain, v_tl_zero, True)
    _, rhs_with_vel = stretch_constraint(bl, tl, 0.85, alpha_gain, v_tl_up, True)

    # If TL moves upward (away from BL), u_BL must keep up → rhs is larger
    assert rhs_with_vel > rhs_no_vel
    assert math.isclose(rhs_with_vel - rhs_no_vel, 0.2, rel_tol=1e-9)


def test_velocity_compensation_opposite_direction():
    """Negative v_j (j approaching i) relaxes the constraint."""
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 0.7])
    v_tl_down = np.array([0.0, 0.0, -0.2])   # TL moving toward BL

    _, rhs_no_vel = stretch_constraint(bl, tl, 0.85, 1.0, np.zeros(3), True)
    _, rhs_closing = stretch_constraint(bl, tl, 0.85, 1.0, v_tl_down, True)

    assert rhs_closing < rhs_no_vel


# ---------------------------------------------------------------------------
# Additional: rhs forces convergence when violated
# ---------------------------------------------------------------------------

def test_rhs_positive_when_violated():
    """When h < 0 (fabric overstretched), rhs > 0 forces u toward j."""
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 1.0])   # dist=1.0 > d_max=0.85 → h=-0.15
    _, rhs = stretch_constraint(bl, tl, 0.85, 1.0, np.zeros(3), True)
    # rhs = -alpha*h = -1*(-0.15) = 0.15 > 0
    assert rhs > 0.0
    assert math.isclose(rhs, 0.15, rel_tol=1e-9)


def test_rhs_negative_when_safe():
    """When h > 0 (fabric within limit), rhs < 0 (constraint easily satisfied)."""
    bl = np.array([0.0, 0.0, 0.0])
    tl = np.array([0.0, 0.0, 0.7])   # dist=0.7, h=0.15
    _, rhs = stretch_constraint(bl, tl, 0.85, 1.0, np.zeros(3), True)
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

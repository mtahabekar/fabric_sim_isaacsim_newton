"""Exact small safety QPs by least-distance programming (no extra dependencies).

    min 0.5 ||u - c||^2   s.t.  A u >= b,  lo <= u <= hi

With z = u - c this is a least-distance program (LDP), solved exactly through
its NNLS dual (Lawson and Hanson, Solving Least Squares Problems, 1974, ch. 23).
Infeasibility is detected exactly. Relaxation is lexicographic (hierarchical
QP, cf. Kanoun et al. 2011; Escande et al. 2014): a small LP finds the least
uniform violation of the hard group under the input box, then of the soft group
with the relaxed hard rows enforced; the LDP then returns the minimum-
intervention command within those relaxed rows.

An infeasible hard group is relaxed by its least violation plus
`hard_tolerance`. Relaxed exactly to the minimum, the feasible set shrinks to
the LP's box vertex: when the followers have little authority over the
violated rows (e.g. a leader stretching the cloth on its own), they were driven
to the input bounds to win a fraction of a unit and kept chattering there,
even after the leaders stopped. The tolerance lets the LDP stay near the
nominal command instead.
"""
import numpy as np
from scipy.optimize import linprog, nnls


def least_distance(g, h, tol=1e-10):
    """min ||x|| s.t. g x >= h. Returns x, or None if the rows are inconsistent."""
    g, h = np.asarray(g, dtype=float), np.asarray(h, dtype=float)
    m, n = g.shape
    if m == 0 or np.all(h <= 0):
        return np.zeros(n)
    scale = np.maximum(np.linalg.norm(g, axis=1), 1e-12)
    g, h = g/scale[:, None], h/scale
    e = np.vstack((g.T, h[None, :]))
    f = np.zeros(n+1)
    f[n] = 1.
    y, _ = nnls(e, f, maxiter=50*(m+n+1))
    r = e@y-f
    if np.linalg.norm(r) <= tol or r[n] >= -tol:
        return None
    x = -r[:n]/r[n]
    # The dual certificate degrades for very large ||x||; verify the primal.
    if np.min(g@x-h) < -1e-7*max(1., np.abs(h).max()):
        return None
    return x


def least_violation(a, b, lower, upper, fixed_rows=None, fixed_rhs=None):
    """min s >= 0 such that a u + s >= b, fixed_rows u >= fixed_rhs, u in box."""
    n = a.shape[1]
    a_ub = [-np.hstack((a, np.ones((len(a), 1))))]
    b_ub = [-b]
    if fixed_rows is not None and len(fixed_rows):
        a_ub.append(-np.hstack((fixed_rows, np.zeros((len(fixed_rows), 1)))))
        b_ub.append(-fixed_rhs)
    result = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack(a_ub), b_ub=np.concatenate(b_ub),
                     bounds=[*zip(lower, upper), (0, None)], method='highs')
    return float(result.x[-1]) if result.status == 0 else None


def safety_qp(center, rows, rhs, groups, lower, upper, tol=1e-7, hard_tolerance=0.):
    """Solve the filter QP; groups[i] is 0 (soft) or 1 (hard) for row i.

    Rows should be dimensionless so a group's uniform slack is meaningful.
    hard_tolerance (row units, >= 0): extra relaxation of an infeasible hard group.
    Returns (u, status, slacks, violation). status: 'feasible',
    'relaxed_soft' / 'relaxed_hard' (most important relaxed group) or
    'solver_failure' (box-clipped center; never reported as safe).
    """
    c = np.asarray(center, dtype=float).ravel()
    a = np.asarray(rows, dtype=float).reshape(-1, len(c))
    b = np.asarray(rhs, dtype=float).ravel()
    groups = np.asarray(groups, dtype=int).ravel()
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if np.any(lower > upper+1e-12):
        raise ValueError('Empty input box')
    n = len(c)
    box = np.vstack((np.eye(n), -np.eye(n)))
    box_rhs = np.concatenate((lower-c, c-upper))
    slacks = np.zeros(2)
    status = 'feasible'
    z = least_distance(np.vstack((a, box)), np.concatenate((b-a@c, box_rhs)))
    if z is None:
        relaxed = b.copy()
        hard, soft = groups == 1, groups == 0
        if hard.any():
            slacks[1] = least_violation(a[hard], b[hard], lower, upper) or 0.
            relaxed[hard] -= slacks[1]+(hard_tolerance if slacks[1] > tol else 0.)
        if soft.any():
            s = least_violation(a[soft], b[soft], lower, upper, a[hard], relaxed[hard])
            slacks[0] = 0. if s is None else s
            relaxed[soft] -= slacks[0]
        # Back off by a relative hair so the LDP sees a strictly feasible set.
        relaxed -= 1e-9*max(1., np.abs(b).max())
        z = least_distance(np.vstack((a, box)), np.concatenate((relaxed-a@c, box_rhs)))
        if z is None:
            z, status = np.clip(c, lower, upper)-c, 'solver_failure'
        else:
            status = ('relaxed_hard' if slacks[1] > tol else 'relaxed_soft' if slacks[0] > tol
                      else 'feasible')
    u = np.clip(c+z, lower, upper)
    violation = float(np.max(b-a@u, initial=0.))
    return u, status, slacks, violation

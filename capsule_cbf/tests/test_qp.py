import unittest
import numpy as np
from scipy.optimize import minimize
from capsule_cbf.qp import least_distance, safety_qp


class SafetyQpTest(unittest.TestCase):
    def test_matches_slsqp_on_random_feasible_problems(self):
        rng = np.random.default_rng(4)
        for _ in range(40):
            a = rng.normal(size=(rng.integers(1, 30), 6))
            u_feasible = rng.normal(scale=.2, size=6)
            b = a@u_feasible-rng.uniform(0, .1, len(a))
            c = rng.normal(scale=.5, size=6)
            lo, hi = np.full(6, -1.), np.full(6, 1.)
            u, status, _, violation = safety_qp(c, a, b, np.zeros(len(a)), lo, hi)
            ref = minimize(lambda x: .5*np.sum((x-c)**2), np.zeros(6), jac=lambda x: x-c,
                           constraints=[{'type': 'ineq', 'fun': lambda x: a@x-b, 'jac': lambda x: a}],
                           bounds=list(zip(lo, hi)), method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
            self.assertEqual(status, 'feasible')
            self.assertLess(violation, 1e-7)
            np.testing.assert_allclose(u, ref.x, atol=1e-5)

    def test_detects_infeasibility(self):
        self.assertIsNone(least_distance(np.array([[1., 0], [-1., 0]]), np.array([1., 1.])))

    def test_hard_group_kept_and_soft_group_relaxed(self):
        # u0 >= 1 (hard) conflicts with -u0 >= 0 (soft): the soft row yields.
        u, status, slacks, _ = safety_qp(np.zeros(6), [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]],
                                         [1., 0.], [1, 0], np.full(6, -2.), np.full(6, 2.))
        self.assertEqual(status, 'relaxed_soft')
        self.assertGreater(u[0], 1-1e-3)
        self.assertGreater(slacks[0], .99)

    def test_rate_box_is_never_violated(self):
        u, status, _, _ = safety_qp(np.zeros(6), [[1, 0, 0, 0, 0, 0]], [5.], [1],
                                    np.full(6, -.05), np.full(6, .05))
        self.assertEqual(status, 'relaxed_hard')
        self.assertLessEqual(u[0], .05+1e-12)

    def test_hard_tolerance_keeps_command_off_the_box_vertex(self):
        # 0.5 u0 >= 1 cannot be met in |u| <= 0.05: least violation 0.975 at u0 = 0.05.
        args = (np.zeros(6), [[.5, 0, 0, 0, 0, 0]], [1.], [1], np.full(6, -.05), np.full(6, .05))
        u, status, slacks, _ = safety_qp(*args)
        self.assertEqual(status, 'relaxed_hard')
        self.assertAlmostEqual(u[0], .05, places=6)
        # One more unit of violation is accepted instead of pinning the command.
        u, status, slacks, violation = safety_qp(*args, hard_tolerance=1.)
        self.assertEqual(status, 'relaxed_hard')
        self.assertAlmostEqual(slacks[1], .975, places=6)
        np.testing.assert_allclose(u, 0., atol=1e-9)
        self.assertAlmostEqual(violation, 1., places=6)

    def test_hard_tolerance_unused_when_feasible(self):
        u, status, _, _ = safety_qp(np.zeros(6), [[1, 0, 0, 0, 0, 0]], [.02], [1],
                                    np.full(6, -.05), np.full(6, .05), hard_tolerance=1.)
        self.assertEqual(status, 'feasible')
        self.assertAlmostEqual(u[0], .02, places=6)


if __name__ == '__main__':
    unittest.main()

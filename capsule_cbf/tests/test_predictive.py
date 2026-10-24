import unittest
import numpy as np
from capsule_cbf.core import Capsule, Settings
from capsule_cbf.strain import TriangleStrain
from capsule_cbf.predictive import PredictiveController, six_direction_sensitivity


class StrainTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        self.strain = TriangleStrain([[0, 1, 2]], [np.eye(2)])

    def test_rigid_motion_invariant(self):
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_allclose(self.strain.measures(self.p@rotation.T+7), 0, atol=1e-12)

    def test_anisotropic_extension_and_signed_shear(self):
        p = self.p.copy()
        p[1, 0] = 1.03
        self.assertAlmostEqual(self.strain.barriers(p)[0, 0], -.01)
        p = self.p.copy()
        p[2, 0] = .3
        m = self.strain.measures(p)[0]
        self.assertAlmostEqual(m[2], .3/np.sqrt(1.09))
        self.assertLess(self.strain.barriers(p)[0, 2], 0)
        p[2, 0] = -.3
        self.assertLess(self.strain.barriers(p)[0, 3], 0)

    def test_rest_metric_handles_diagonal_triangle(self):
        p = self.p[[1, 2, 0]]
        dm = np.column_stack((p[1, :2]-p[0, :2], p[2, :2]-p[0, :2]))
        strain = TriangleStrain([[0, 1, 2]], [np.linalg.inv(dm)])
        np.testing.assert_allclose(strain.measures(p), 0, atol=1e-12)

    def test_local_extension_hidden_by_unchanged_corners(self):
        points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [.5, .5, 1.]], float)
        tri = [[0, 1, 4]]
        rest = np.column_stack((points[1, :2]-points[0, :2], points[4, :2]-points[0, :2]))
        strain = TriangleStrain(tri, [np.linalg.inv(rest)])
        points[4, 2] += .2
        self.assertLess(strain.barriers(points).min(), 0)


class PredictiveTests(unittest.TestCase):
    def setup_controller(self):
        points = np.array([[-.5, -.5, 1.], [.5, -.5, 1.], [-.5, .5, 1.], [.5, .5, 1.]])
        strain = TriangleStrain([[0, 1, 2]], [np.eye(2)])
        cfg = Settings(prediction_iterations=3, strain_enabled=True, vertical_formation_recovery=True)
        ctrl = PredictiveController(Capsule([0, 0, .16], [.22, .22, .16]),
                                    strain, points, .008, cfg, 2)
        return ctrl, points

    def test_six_direction_derivative_against_independent_direction(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(2, 9, 6))
        rollout = lambda u, leaders: np.einsum('ijk,k->ij', a, u.ravel())+2
        u, v = np.zeros((2, 3)), np.ones((2, 3))
        jac = six_direction_sensitivity(rollout, u, v, .02)
        direction = rng.normal(size=6)
        actual = (rollout(u+.01*direction.reshape(2, 3), v)-rollout(u, v))/.01
        np.testing.assert_allclose(jac@direction, actual, atol=1e-11)

    def test_safe_nominal_uses_one_rollout_and_no_corner_stretch_proxy(self):
        ctrl, points = self.setup_controller()
        h = ctrl.barriers(points)
        rollout = lambda u, v: np.tile(h, (2, 1))
        leaders = np.array([[1., 0., 0.], [1., 0., 0.]])
        u, debug = ctrl.update(points, points, leaders, 1/60, rollout)
        np.testing.assert_allclose(u, leaders)
        self.assertEqual(debug['status'], 'feasible')
        self.assertEqual(debug['rollout_count'], 1)

    def test_obstacle_and_strain_correction_checked_together(self):
        ctrl, points = self.setup_controller()
        h = ctrl.barriers(points)
        def rollout(u, leaders):
            pred = np.tile(h, (2, 1))
            pred[:, 0] = h[0]*np.exp(-2*np.arange(1, 3)/120)-.002+.02*u[0, 2]
            pred[:, 4] = -.002+.05*u[1, 0]
            return pred
        u, debug = ctrl.update(points, points, np.zeros((2, 3)), 1/60, rollout)
        self.assertEqual(debug['status'], 'feasible')
        self.assertGreater(u[1, 0], .4)
        self.assertGreater(u[0, 2], .09)
        self.assertGreater(debug['rollout_count'], 1)
        self.assertLessEqual(debug['max_violation'], 1e-6)

    def test_infeasible_pinned_leader_not_declared_safe(self):
        ctrl, points = self.setup_controller()
        def rollout(u, leaders):
            pred = np.tile(ctrl.barriers(points), (2, 1))
            pred[:, 2] = -.1
            return pred
        _, debug = ctrl.update(points, points, np.zeros((2, 3)), 1/60, rollout)
        self.assertEqual(debug['status'], 'prediction_not_safe')
        self.assertGreater(debug['max_violation'], .1)

    def test_dynamics_interface_checks_clearance_and_strain_without_probes(self):
        from capsule_cbf.cloth_state import ClothState
        ctrl, points = self.setup_controller()
        ctrl.settings.sensitivity_method = 'solver_tangent'
        # Allow the synthetic strain repair in one trust-region step.
        ctrl.settings.prediction_trust_displacement = .02
        h = ctrl.barriers(points)
        def rollout(u, leaders):
            pred = np.tile(h, (2, 1))
            pred[:, 0] = h[0]*np.exp(-2*np.arange(1, 3)/120)-.002+.02*u[0, 2]
            pred[:, 4] = -.002+.05*u[1, 0]
            return pred
        def sensitivity(u, leaders):
            jac = np.zeros((2, len(h), 6))
            jac[:, 0, 2], jac[:, 4, 3] = .02, .05
            return jac
        rollout.sensitivity = sensitivity
        for _ in range(2):
            state = ClothState(points, np.zeros_like(points))
            _, debug = ctrl.update(state, points, np.zeros((2, 3)), 1/60, rollout)
            self.assertTrue(debug['state_velocity_available'])
            self.assertEqual(debug['status'], 'feasible')
            self.assertEqual(debug['rollout_count'], 2)
            self.assertEqual(debug['sensitivity_modes'], ['solver_tangent'])
            self.assertFalse(debug['sensitivity_reused'])

    def test_unsafe_intermediate_substep_rejected(self):
        ctrl, points = self.setup_controller()
        def rollout(u, leaders):
            pred = np.tile(ctrl.barriers(points), (2, 1))
            pred[0, 0] = -.02
            return pred
        _, debug = ctrl.update(points, points, np.zeros((2, 3)), 1/60, rollout)
        self.assertEqual(debug['status'], 'prediction_not_safe')

    def test_nonfinite_prediction_rejected(self):
        ctrl, points = self.setup_controller()
        with self.assertRaises(ValueError):
            ctrl.update(points, points, np.zeros((2, 3)), 1/60,
                        lambda u, v: np.full((2, 8), np.nan))

    def test_linear_feasibility_does_not_override_nonlinear_check(self):
        ctrl, points = self.setup_controller()
        ctrl.settings.prediction_iterations = 1
        h = ctrl.barriers(points)
        def rollout(u, leaders):
            pred = np.tile(h, (2, 1))
            x = u[0, 0]
            pred[:, 4] = h[4]-.002+.02*x-.2*x*x
            return pred
        _, debug = ctrl.update(points, points, np.zeros((2, 3)), 1/60, rollout)
        self.assertEqual(debug['linear_status'], 'feasible')
        self.assertEqual(debug['status'], 'prediction_not_safe')
        self.assertGreater(debug['max_strain_violation'], 0.)


if __name__ == '__main__':
    unittest.main()

import itertools
import unittest
import numpy as np
from capsule_cbf.core import Capsule, CapsuleController, Settings, material_weights, solve_filter


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.cap = Capsule([0., 0., .16], [.22, .22, .16])

    def test_box_enclosure_and_surface(self):
        corners = np.array(list(itertools.product((-.22, .22), (-.22, .22), (0., .32))))
        self.assertTrue(np.all(self.cap.distances(corners)[0] <= 1e-12))
        np.testing.assert_allclose(self.cap.distances(self.cap.cells)[0], 0., atol=1e-12)
        self.assertLess(self.cap.distances([[0., 0., .16]])[0][0], 0.)

    def test_centered_sphere_top_encloses_box_and_lowers_peak(self):
        cap = Capsule([0, 0, .16], [.22, .22, .16], top_cap_mode='centered_sphere')
        radius = np.sqrt(2*.22**2+.16**2)
        self.assertAlmostEqual(cap.top_sphere_radius, radius)
        peak = cap.surface([1.], [0.])[0]
        np.testing.assert_allclose(peak, [0, 0, .16+radius], atol=1e-12)
        self.assertLess(peak[2], self.cap.surface([1.], [0.])[0, 2])
        corners = np.array(list(itertools.product((-.22, .22), (-.22, .22), (0., .32))))
        self.assertTrue(np.all(cap.distances(corners)[0] <= 1e-12))
        np.testing.assert_allclose(cap.distances(cap.cells)[0], 0., atol=1e-12)
        top = cap.cells[cap.cells[:, 2] > .32]
        np.testing.assert_allclose(np.linalg.norm(top-cap.center, axis=1), radius)

    def test_centered_top_signed_distance_normals_and_projection(self):
        cap = Capsule([.1, -.2, .16], [.22, .22, .16], top_cap_mode='centered_sphere')
        points = np.random.default_rng(7).uniform(-.7, .7, (300, 3))+cap.center
        d, normals, surface = cap.distances(points)
        np.testing.assert_allclose(cap.distances(surface)[0], 0., atol=1e-12)
        np.testing.assert_allclose(points-surface, d[:, None]*normals, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-12)
        for axis in range(3):
            eps = np.eye(3)[axis]*1e-6
            numerical = (cap.distances(points+eps)[0]-cap.distances(points-eps)[0])/2e-6
            np.testing.assert_allclose(numerical, normals[:, axis], atol=1e-7)
        np.testing.assert_allclose(cap.distances(points, .008)[0], d-.008)

    def test_gradient_finite_difference(self):
        points = np.array([[.5, .1, .2], [.1, .1, .7], [.1, .2, -.3]])
        d, n, _ = self.cap.distances(points, .008)
        for axis in range(3):
            step = np.eye(3)[axis]*1e-6
            numerical = (self.cap.distances(points+step, .008)[0] -
                         self.cap.distances(points-step, .008)[0]) / 2e-6
            np.testing.assert_allclose(numerical, n[:, axis], atol=1e-7)

    def test_material_row_derivative_with_leaders(self):
        rest = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1], [.1, -.2, .7]])
        weights = material_weights(rest)
        np.testing.assert_allclose(weights[:4], np.eye(4))
        commands = np.arange(12).reshape(4, 3)*.013
        normals = self.cap.distances(rest)[1]
        expected = np.einsum('ij,ij->i', normals, weights@commands)
        eps = 1e-6
        actual = (self.cap.distances(rest+eps*(weights@commands))[0]-
                  self.cap.distances(rest-eps*(weights@commands))[0])/(2*eps)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_multiple_rows_and_speed_ball(self):
        rows = np.zeros((3, 6))
        rows[0, 0], rows[1, 1], rows[2, 3] = 1, 1, -1
        u, status, violation = solve_filter(np.zeros(6), rows, [.08, .06, .09], .2)
        self.assertEqual(status, 'feasible')
        self.assertLess(violation, 1e-6)
        np.testing.assert_allclose(u.ravel()[[0, 1, 3]], [.08, .06, -.09], atol=1e-5)
        self.assertTrue(np.all(np.linalg.norm(u, axis=1) <= .2+1e-9))

    def test_infeasible_not_reported_safe(self):
        rows = np.zeros((2, 6))
        rows[:, 0] = [1, -1]
        u, status, violation = solve_filter(np.zeros(6), rows, [.1, .1], .2)
        self.assertEqual(status, 'recovery_not_safe')
        self.assertGreater(violation, .09)
        self.assertTrue(np.all(np.linalg.norm(u, axis=1) <= .2+1e-9))

    def test_unlimited_nominal_and_safety_correction(self):
        nominal = np.array([[1., 0., 0.], [0., -1.5, 0.]])
        u, status, violation = solve_filter(nominal, [], [])
        self.assertEqual(status, 'feasible')
        np.testing.assert_allclose(u, nominal)
        row = np.zeros((1, 6))
        row[0, 2] = 1.
        u, status, violation = solve_filter(nominal, row, [2.])
        self.assertEqual(status, 'feasible')
        self.assertLess(violation, 1e-6)
        self.assertAlmostEqual(u[0, 2], 2., places=5)
        np.testing.assert_allclose(u[:, :2], nominal[:, :2])

    def test_unlimited_recovery_keeps_obstacle_rows_soft(self):
        rows = np.zeros((3, 6))
        rows[0, 0], rows[1, 0], rows[2, 1] = 1., -1., 1.
        u, status, violation = solve_filter(np.zeros(6), rows, [.1, .1, 1.])
        self.assertEqual(status, 'recovery_not_safe')
        self.assertGreater(violation, .09)
        self.assertTrue(np.isfinite(u).all())
        self.assertGreater(u[0, 1], .99)

    def test_large_working_set_checks_all_rows(self):
        rng = np.random.default_rng(17)
        rows = rng.normal(size=(16000, 6))
        feasible = np.array([.1, -.04, .08, -.1, .07, .02])
        rhs = rows@feasible-.01
        u, status, violation = solve_filter(np.zeros(6), rows, rhs)
        self.assertEqual(status, 'feasible')
        self.assertLessEqual(violation, 1e-6)
        self.assertTrue(np.all(rows@u.ravel() >= rhs-1e-6))

    def test_large_infeasible_rows_do_not_report_safe(self):
        rows = np.zeros((16000, 6))
        rows[:8000, 0], rows[8000:, 0] = 1., -1.
        _, status, violation = solve_filter(np.zeros(6), rows, np.full(16000, .1))
        self.assertEqual(status, 'recovery_not_safe')
        self.assertGreater(violation, .09)

    def controller(self, z=.72):
        x, y = np.meshgrid(np.linspace(-.42, .42, 15), np.linspace(-.42, .42, 15))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
        corners = points[[0, 14, 210, 224]]
        return CapsuleController(self.cap, points, corners, .008), points, corners

    def test_unlimited_controller_tracks_fast_leaders(self):
        controller, points, corners = self.controller(z=3.)
        self.assertEqual(controller.settings.max_speed, 0.)
        leaders = np.array([[1., 0., 0.], [1., 0., 0.]])
        u, debug = controller.update(points, corners, leaders, 1/60)
        self.assertEqual(debug['status'], 'feasible')
        np.testing.assert_allclose(u, leaders, atol=1e-6)

    def test_basin_seam_and_distinct_interactions(self):
        controller, _, _ = self.controller()
        # Same meridian band: azimuth 0 and 23 are adjacent.
        groups = controller.basins({120: .1, 143: .1, 132: .03})
        self.assertEqual(len(groups), 2)
        self.assertTrue(any(set(cells) == {120, 143} for cells, _ in groups))

    def test_empty_directions_and_invisible_cells(self):
        controller, _, _ = self.controller()
        self.assertEqual(controller.basins({5: .02}), [([5], [5])])
        self.assertEqual(controller.basins({}), [])

    def test_follower_output_changes_and_satisfies_rows(self):
        controller, points, corners = self.controller(z=.70)
        leaders = np.array([[0., 0., -.08], [0., 0., -.08]])
        u, debug = controller.update(points, corners, leaders, 1/60)
        self.assertEqual(debug['status'], 'feasible')
        self.assertGreater(debug['correction'], .01)
        self.assertGreater(debug['basin_rows'], 0)
        self.assertGreater(debug['total_rows'], debug['basin_rows']+3)
        self.assertGreater(np.mean(u[:, 2]), -.08)
        self.assertTrue(np.all(debug['rows']@u.ravel() >= debug['rhs']-1e-6))

    def test_synthetic_closed_loop(self):
        controller, points, corners = self.controller()
        # Ideal bilinear plant only: this verifies integration/sign, not Newton dynamics.
        min_h = np.inf
        max_correction = 0.
        for _ in range(90):
            leaders = np.array([[0., 0., -.025], [0., 0., -.025]])
            u, debug = controller.update(points, corners, leaders, 1/60)
            self.assertEqual(debug['status'], 'feasible')
            command = np.vstack((u, leaders))
            points += (controller.weights@command)/60
            corners += command/60
            min_h = min(min_h, self.cap.distances(points, .008)[0].min()-.05)
            max_correction = max(max_correction, debug['correction'])
        self.assertGreaterEqual(min_h, -1e-5)
        self.assertGreater(max_correction, .001)

    def test_invalid_state_rejected(self):
        controller, points, corners = self.controller()
        points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            controller.update(points, corners, np.zeros((2, 3)), 1/60)


if __name__ == '__main__':
    unittest.main()

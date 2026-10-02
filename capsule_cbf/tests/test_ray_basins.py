import unittest
import numpy as np
from capsule_cbf.core import Capsule, Settings
from capsule_cbf.ray_basins import basin_representatives, RayBasinController


class RayBasinTests(unittest.TestCase):
    def test_knn_separates_distant_patches_and_keeps_worst_hit(self):
        p = np.array([[0, 0, 0], [.01, 0, 0], [.02, 0, 0], [1, 0, 0], [1.01, 0, 0]])
        chosen = basin_representatives(p, np.array([.03, -.01, .02, .01, -.02]), 4, .08)
        np.testing.assert_array_equal(chosen, [1, 4])
        self.assertEqual(len(basin_representatives(np.empty((0, 3)), [], 4, .08)), 0)
        np.testing.assert_array_equal(basin_representatives(p[:1], [.1], 4, .08), [0])

    def fixture(self, hit_points):
        cap = Capsule([0, 0, 0], [.1, .1, .1])
        cfg = Settings(prediction_iterations=2)
        class Detector:
            def detect(self):
                return dict(points=hit_points, faces=np.arange(len(hit_points)),
                            weights=np.tile([1., 0, 0], (len(hit_points), 1)))
        corners = np.array([[-.4, -.4, 1], [.4, -.4, 1], [-.4, .4, 1], [.4, .4, 1]])
        ctrl = RayBasinController(cap, None, corners, 0, cfg, 3, Detector())
        class Rollout:
            def select_hits(self, faces, weights):
                self.faces = faces
            def __call__(self, u, leaders):
                self.last = u.copy()
                h = cap.distances(hit_points[self.faces])[0]-cfg.margin
                # Vertical follower motion improves clearance at every substep.
                return h[None, :]+np.arange(1, 4)[:, None]/180*u[:, 2].mean()
        return ctrl, Rollout(), corners

    def test_no_hits_skips_prediction_and_no_vertical_drift(self):
        ctrl, rollout, corners = self.fixture(np.empty((0, 3)))
        corners[:2, 2] -= .2
        u, debug = ctrl.update(corners, corners, np.zeros((2, 3)), 1/60, rollout)
        np.testing.assert_array_equal(u, 0)
        self.assertEqual(debug['rollout_count'], 0)
        self.assertEqual(debug['total_rows'], 0)
        leaders = np.full((2, 3), 5.)
        u, _ = ctrl.update(corners, corners, leaders, 1/60, rollout)
        np.testing.assert_array_equal(u, leaders)  # no inherited speed bound

    def test_only_inside_margin_hits_selected_and_one_row_per_basin(self):
        ctrl, rollout, corners = self.fixture(np.array([[0, 0, .28], [.005, 0, .279], [0, 0, 1.]]))
        u, debug = ctrl.update(corners, corners, np.zeros((2, 3)), 1/60, rollout)
        self.assertEqual(debug['occupied_cells'], 2)
        self.assertEqual(debug['basin_count'], 1)
        self.assertEqual(debug['total_rows'], 1)
        self.assertEqual(debug['status'], 'feasible')
        self.assertGreater(u[:, 2].mean(), 0)
        self.assertLessEqual(debug['max_violation'], 1e-6)


if __name__ == '__main__':
    unittest.main()

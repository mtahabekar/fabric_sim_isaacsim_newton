import json
import math
import unittest
import numpy as np
from capsule_cbf.sensitivity_cbf import ConformalBound, local_minima, rest_rings

try:
    import warp as wp
    import newton  # noqa: F401
except ImportError:
    wp = None

DEBUG_ARRAYS = ('surfaces', 'rows', 'rhs', 'detected_indices', 'basin_indices', 'detected_points',
                'basin_points', 'candidate_indices')


class PureHelpersTest(unittest.TestCase):
    def test_rings_and_local_minima_on_grid(self):
        n = 6
        tris = []
        for j in range(n-1):
            for i in range(n-1):
                a = j*n+i
                tris += [(a, a+1, a+n), (a+1, a+n+1, a+n)]
        table = rest_rings(np.array(tris), n*n, 1)
        self.assertEqual(set(table[0].tolist()), {0, 1, n})
        values = np.ones(n*n)
        values[[7, 28]] = 0
        self.assertEqual(local_minima(values, rest_rings(np.array(tris), n*n, 2)).tolist(), [7, 28])

    def test_conformal_rank_and_fallback(self):
        bound = ConformalBound(1, 100, .9, fallback=.5, minimum=20)
        bound.add(0, np.arange(10.))
        self.assertEqual(bound.bound(0), .5)
        bound.add(0, np.arange(10., 49.))
        rank = math.ceil(50*.9)  # n=49
        self.assertEqual(bound.bound(0), float(rank-1))


@unittest.skipIf(wp is None, 'Newton/Warp unavailable')
class ControllerOnNewtonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        from capsule_cbf.core import Settings
        from capsule_cbf.scenario import build_plant
        cls.settings = Settings(prediction_grid_cells=8, strain_enabled=True, max_speed=.5)

        def settled(z):
            # The affine model is linearized about tensioned cloth; a freshly
            # spawned flat sheet is slack and free-falling (see report).
            plant = build_plant('cpu', grid=16, cloth_z=z, capture=False)
            for _ in range(60):
                plant.step(np.zeros((2, 3)), np.zeros((2, 3)))
            return plant
        cls.build = staticmethod(settled)

    def controller(self, plant):
        from capsule_cbf.reduced_cloth import ReducedCloth
        from capsule_cbf.scenario import FRAME_DT
        from capsule_cbf.sensitivity_cbf import SensitivityCbfController
        cfg = self.settings
        red = ReducedCloth(plant.model, plant.patch, plant.corner_ids, plant.offsets, FRAME_DT,
                           cells=cfg.prediction_grid_cells, stride=cfg.horizon_stride)
        return SensitivityCbfController(plant.capsule, red, plant.rest[plant.corners], .008, cfg, FRAME_DT)

    def step(self, plant, ctrl, leaders, nominal=None):
        from capsule_cbf.cloth_state import ClothState
        q, v = plant.state()
        command, debug = ctrl.update(ClothState(q, v), q[plant.corners], leaders, 1/60, nominal=nominal)
        plant.step(command, leaders)
        return command, debug

    def test_far_from_limits_passes_nominal_and_reports_json(self):
        plant = self.build(.95)
        ctrl = self.controller(plant)
        for _ in range(3):
            command, debug = self.step(plant, ctrl, np.zeros((2, 3)))
        self.assertEqual(debug['status'], 'feasible')
        self.assertEqual(debug['candidate_count'], 0)
        self.assertLess(abs(command).max(), 1e-3)
        json.dumps({k: v for k, v in debug.items() if k not in DEBUG_ARRAYS}, allow_nan=False)

    def test_overstretching_task_command_is_reduced(self):
        plant = self.build(.95)
        ctrl = self.controller(plant)
        pull = np.array([[0., -.2, 0.], [0., -.2, 0.]])
        for _ in range(30):
            command, debug = self.step(plant, ctrl, np.zeros((2, 3)), nominal=pull)
        self.assertLess(max(debug['tension']), 1.2*self.settings.tension_limit)
        self.assertGreater(command[:, 1].min(), -.05)

    def test_conflicting_descent_keeps_stretch_priority_and_reports_it(self):
        # Starting 7 cm from the margin while leaders descend, avoiding the
        # capsule would require stretching the cloth: a physical conflict.
        # Stretch has priority; the yielded clearance must be reported.
        plant = self.build(.66)
        ctrl = self.controller(plant)
        leaders = np.array([[0., 0., -.1], [0., 0., -.1]])
        engaged, tension, statuses = None, [], set()
        for frame in range(60):
            command, debug = self.step(plant, ctrl, leaders)
            tension.append(max(debug['tension']))
            statuses.add(debug['status'])
            if engaged is None and debug['candidate_count']:
                engaged = frame
        self.assertIsNotNone(engaged)
        self.assertLess(max(tension), 1.2*self.settings.tension_limit)
        self.assertTrue(statuses & {'relaxed_soft', 'relaxed_hard'})


if __name__ == '__main__':
    unittest.main()

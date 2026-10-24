"""Coarse dynamics must retain measured geometry, mass, and device derivatives."""
import unittest
from dataclasses import replace
import numpy as np
try:
    import newton
    import warp as wp
except ImportError:
    newton = None
if newton is not None:
    wp.config.kernel_cache_dir = '/tmp/capsule-cbf-warp-cache'


@unittest.skipIf(newton is None, 'Newton/Warp runtime unavailable')
class CoarsePredictionTests(unittest.TestCase):
    def fixture(self, cells=2, backend='warp'):
        from capsule_cbf.tests.test_newton_runtime import NewtonRuntimeTests
        from capsule_cbf.coarse_prediction import CoarseNewtonPredictor
        model, live, fine, ids, starts, cfg, cap = NewtonRuntimeTests().fixture(
            sensitivity_method='solver_tangent', top_cap_mode='centered_sphere')
        cfg = replace(cfg, ray_basins_enabled=True, strain_enabled=False,
                      prediction_grid_cells=cells, sensitivity_backend=backend)
        coarse = CoarseNewtonPredictor(model, ids, np.arange(4), np.zeros((4, 3)),
                                       3, 1/180, 5, 8, cap, .008, cfg, capture=False)
        return model, live, fine, coarse, starts, cap

    def test_material_mapping_mass_and_boundaries(self):
        from capsule_cbf.coarse_prediction import MaterialGrid
        model, live, fine, pred, starts, cap = self.fixture()
        self.assertEqual(model.particle_count, 25)
        self.assertEqual(pred.runtime.model.particle_count, 9)
        rng = np.random.default_rng(19)
        uv = rng.uniform(-.06, .06, (50, 2))
        faces, w = pred.grid.locate(uv)
        mapped = np.einsum('ij,ijk->ik', w, pred.grid.vertices[pred.grid.triangles[faces]])
        np.testing.assert_allclose(mapped[:, :2], uv, atol=1e-8)
        self.assertAlmostEqual(float(model.particle_mass.numpy().sum()),
                               float(pred.runtime.model.particle_mass.numpy().sum()), places=6)
        for name in ('shape_transform', 'shape_scale', 'shape_flags', 'shape_material_mu', 'shape_material_ke'):
            np.testing.assert_array_equal(getattr(model, name).numpy(), getattr(pred.runtime.model, name).numpy())
        # Nonzero-width patches insert boundary lines even at a coarse base resolution.
        rest = np.array([[x, y, .5] for y in range(9) for x in range(9)], dtype=float)
        patch, ids = [], []
        for cid, (cx, cy) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
            indices = np.flatnonzero(np.all(abs(rest[:, :2]-[cx, cy]) <= 1, axis=1))
            patch.extend(indices)
            ids.extend([cid]*len(indices))
        grid = MaterialGrid(rest, np.array(patch), np.array(ids), 2)
        np.testing.assert_array_equal(grid.x, [0, 1, 4, 7, 8])
        np.testing.assert_array_equal(grid.y, grid.x)

    def test_state_restriction_residual_and_frame_reset(self):
        model, live, fine, pred, starts, cap = self.fixture()
        q = live.state.particle_q.numpy()
        q[6, 2] -= .025  # wrinkle at a vertex absent from the coarse mesh
        v = np.random.default_rng(4).normal(0, .01, q.shape).astype(np.float32)
        live.state.particle_q.assign(q)
        live.state.particle_qd.assign(v)
        faces = np.array([2, 10, 19])
        weights = np.tile([.2, .3, .5], (3, 1))
        pred.begin(live.state, starts)
        pred.select_hits(faces, weights)
        np.testing.assert_array_equal(pred.current_cloth_state().positions, q)
        np.testing.assert_array_equal(pred.snapshot.particle_q.numpy(), q[pred.grid.source])
        np.testing.assert_array_equal(pred.snapshot.particle_qd.numpy(), v[pred.grid.source])
        r = pred.runtime
        r._observe(pred.snapshot, 0)
        actual_hits = np.einsum('ij,ijk->ik', weights, q[model.tri_indices.numpy()[faces]])
        expected = cap.distances(actual_hits, .008)[0]-r.settings.margin
        np.testing.assert_allclose(r.hit_observations.numpy()[0, :3], expected, atol=2e-7)
        u = np.full((2, 3), .01)
        value = pred(u, u)
        jac = pred.sensitivity(u, u)
        pred(u+.02, u)
        np.testing.assert_array_equal(pred(u, u), value)
        np.testing.assert_array_equal(pred.sensitivity(u, u), jac)
        np.testing.assert_array_equal(live.state.particle_q.numpy(), q)
        np.testing.assert_array_equal(live.state.particle_qd.numpy(), v)
        q[:, 2] += .01
        live.state.particle_q.assign(q)
        pred.begin(live.state, starts)
        with self.assertRaisesRegex(RuntimeError, 'Select fine-mesh'):
            pred(u, u)
        pred.select_hits(faces, weights)
        r._observe(pred.snapshot, 0)
        actual_hits[:, 2] += .01
        np.testing.assert_allclose(r.hit_observations.numpy()[0, :3],
                                   cap.distances(actual_hits, .008)[0]-r.settings.margin, atol=2e-7)

    def test_residual_hit_derivative_and_cpu_reference(self):
        results = []
        for backend in ('warp', 'cpu'):
            model, live, fine, pred, starts, cap = self.fixture(backend=backend)
            q = live.state.particle_q.numpy()
            q[6, :2] += [.003, -.002]
            q[6, 2] -= .01
            live.state.particle_q.assign(q)
            live.state.particle_qd.assign(np.random.default_rng(6).normal(0, .03, q.shape))
            faces = np.array([2, 10, 19])
            pred.begin(live.state, starts)
            pred.select_hits(faces, np.tile([.2, .3, .5], (3, 1)))
            u, leaders = np.full((2, 3), .01), np.zeros((2, 3))
            value = pred(u, leaders)
            jac = pred.sensitivity(u, leaders)
            results.append(jac)
            central = np.stack([(pred(u+.01*d.reshape(2, 3), leaders)
                                 -pred(u-.01*d.reshape(2, 3), leaders))/.02 for d in np.eye(6)], axis=-1)
            np.testing.assert_allclose(jac, central, atol=3e-5, rtol=.02)
        np.testing.assert_allclose(results[0], results[1], atol=2e-6, rtol=.002)

    def test_identity_resolution_matches_original_solver(self):
        model, live, fine, pred, starts, cap = self.fixture(cells=4)
        faces, weights = np.array([0, 4, 21]), np.tile([.2, .3, .5], (3, 1))
        u, leaders = np.full((2, 3), .01), np.full((2, 3), -.01)
        outputs = []
        for predictor in (fine, pred):
            predictor.begin(live.state, starts)
            predictor.select_hits(faces, weights)
            outputs.append((predictor(u, leaders), predictor.final_points(), predictor.sensitivity(u, leaders)))
        for a, b in zip(*outputs):
            np.testing.assert_allclose(a, b, atol=3e-6, rtol=.003)

    def test_cuda_graph_with_fine_live_mesh(self):
        if not wp.is_cuda_available():
            self.skipTest('CUDA device unavailable')
        from capsule_cbf.core import Settings
        from capsule_cbf.validate_newton import build_fixture
        cfg = Settings(prediction_grid_cells=2, sensitivity_method='solver_tangent', top_cap_mode='centered_sphere')
        model, live, pred, ctrl, corners, starts = build_fixture(8, 3, 'cuda:0', cfg, .55)
        before = live.state.particle_q.numpy()
        self.assertLess(pred.runtime.model.particle_count, model.particle_count)
        self.assertIsNotNone(pred.runtime.graph)
        u, leaders = np.full((2, 3), .01), np.zeros((2, 3))
        pred.begin(live.state, starts)
        pred.select_hits(np.array([0, 16, 71]), np.tile([.2, .3, .5], (3, 1)))
        value = pred(u, leaders)
        jac = pred.sensitivity(u, leaders)
        direction = np.arange(6).reshape(2, 3)/5
        central = (pred(u+.01*direction, leaders)-pred(u-.01*direction, leaders))/.02
        np.testing.assert_allclose(jac@direction.ravel(), central, atol=3e-5, rtol=.02)
        np.testing.assert_allclose(pred(u, leaders), value, atol=2e-6)
        self.assertEqual(pred.final_points().shape, before.shape)
        np.testing.assert_array_equal(before, live.state.particle_q.numpy())
        pred.begin(live.state, starts)
        command, debug = ctrl.update(pred.current_cloth_state(), before[corners], leaders, 1/60, pred)
        self.assertTrue(np.isfinite(command).all())
        self.assertEqual(debug['prediction_model'], 'coarse')
        self.assertEqual(debug['state_particle_count'], model.particle_count)


if __name__ == '__main__':
    unittest.main()

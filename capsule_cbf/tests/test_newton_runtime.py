"""Actual Newton checks; skip on host Python without Newton/Warp.

Use the Isaac Python runtime with Newton and Warp on PYTHONPATH; no Kit/ROS
startup is necessary. CPU fixtures are intentionally small, not FPS benchmarks.
"""
import unittest
import numpy as np
try:
    import newton
    import warp as wp
except ImportError:
    newton = None
if newton is not None:
    # Set this before the CUDA availability decorator can initialize Warp.
    wp.config.kernel_cache_dir = '/tmp/capsule-cbf-warp-cache'


@unittest.skipIf(newton is None, 'Newton/Warp runtime unavailable')
class NewtonRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = '/tmp/capsule-cbf-warp-cache'
        wp.init()

    def fixture(self, substeps=3, contact=False):
        from newton.solvers import style3d
        from capsule_cbf.core import Capsule, Settings
        from capsule_cbf.newton_runtime import ClothRuntime, NewtonPredictor
        with wp.ScopedDevice('cpu'):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            builder.add_ground_plane()
            builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0., 0., .16), wp.quat_identity()),
                                  hx=.22, hy=.22, hz=.16)
            style3d.add_cloth_grid(builder, pos=wp.vec3(-.06, -.06, .34 if contact else 1.05),
                                  rot=wp.quat_identity(), vel=wp.vec3(),
                                  dim_x=4, dim_y=4, cell_x=.03, cell_y=.03, mass=.003,
                                  tri_aniso_ke=wp.vec3(3.e4, 3.e4, 6.e3),
                                  edge_aniso_ke=wp.vec3(.002, .001, .0005), particle_radius=.008)
            rest = np.asarray(builder.particle_q, dtype=np.float32)
            wanted = np.array([[-.06, -.06], [.06, -.06], [-.06, .06], [.06, .06]])
            indices = np.array([np.argmin(np.linalg.norm(rest[:, :2]-p, axis=1)) for p in wanted])
            for i in indices:
                builder.particle_flags[int(i)] &= ~int(newton.ParticleFlags.ACTIVE)
            model = builder.finalize(device='cpu')
            model.set_gravity((0, 0, -9.81))
            model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 4000., .01, 2.
            model.soft_contact_margin = .012
            live = ClothRuntime(model, indices, np.arange(4), np.zeros((4, 3)),
                                substeps, 1/60/substeps, 5, 8, capture=False)
            if contact:
                q = live.state.particle_q.numpy()
                q[12, 2] = .318
                live.state.particle_q.assign(q)
            cfg = Settings(strain_enabled=True, ray_basins_enabled=False)
            cap = Capsule([0, 0, .16], [.22, .22, .16])
            predictor = NewtonPredictor(model, indices, np.arange(4), np.zeros((4, 3)),
                                         substeps, 1/60/substeps, 5, 8, cap, .008, cfg, capture=False)
        return model, live, predictor, indices, rest[indices], cfg, cap

    def test_substep_boundary_velocity_and_odd_even_rollout_agreement(self):
        from capsule_cbf.strain import TriangleStrain
        for substeps in (3, 4):
            model, live, pred, ids, starts, cfg, cap = self.fixture(substeps)
            commands = np.array([[.12, -.05, .02], [-.07, .03, -.02], [.08, .01, .04], [.06, 0, .03]])
            pred.begin(live.state, starts)
            observed = pred(commands[:2], commands[2:])
            expected = pred.final_points()
            live.advance(starts, commands)
            np.testing.assert_allclose(live.state.particle_q.numpy(), expected, atol=1e-7)
            np.testing.assert_allclose(expected[ids], starts+commands/60, atol=1e-7)
            np.testing.assert_allclose(live.state.particle_qd.numpy()[ids], commands, atol=1e-7)
            strain = TriangleStrain(model.tri_indices.numpy(), model.tri_poses.numpy())
            np.testing.assert_allclose(observed[-1, len(expected):].reshape(-1, 4),
                                       strain.barriers(expected), atol=3e-6)
            for step in range(substeps):
                pinned = starts+commands*(step+1)/60/substeps
                h = cap.distances(pinned, .008)[0]-cfg.margin
                np.testing.assert_allclose(observed[step, ids], h, atol=2e-7)

    def test_ray_nearest_surface_and_barycentric_support(self):
        from types import SimpleNamespace
        from capsule_cbf.core import Capsule
        from capsule_cbf.newton_runtime import CapsuleRayDetector
        # Two sheets along each upward ray: retain the nearer sheet, not both.
        points = np.array([[-2, -2, .5], [2, -2, .5], [0, 2, .5],
                           [-2, -2, 1.], [2, -2, 1.], [0, 2, 1.]], np.float32)
        tris = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
        model = SimpleNamespace(device='cpu', tri_indices=wp.array(tris, dtype=int, device='cpu'))
        state = SimpleNamespace(particle_q=wp.array(points, dtype=wp.vec3, device='cpu'))
        detector = CapsuleRayDetector(model, state, Capsule([0, 0, 0], [.1, .1, .1]))
        hits = detector.detect()
        self.assertGreater(len(hits['faces']), 0)
        np.testing.assert_array_equal(hits['faces'], 0)
        expected = np.einsum('ij,ijk->ik', hits['weights'], points[tris[hits['faces']]])
        np.testing.assert_allclose(expected, hits['points'], atol=1e-6)
        np.testing.assert_allclose(hits['points'][:, 2], .5, atol=1e-6)
        points[:3, 2] = .6
        state.particle_q.assign(points)
        np.testing.assert_allclose(detector.detect()['points'][:, 2], .6, atol=1e-6)

    def test_selected_material_hits_match_full_particle_observations(self):
        _, live, pred, _, starts, cfg, cap = self.fixture()
        faces, weights = np.array([0, 10]), np.array([[.2, .3, .5], [0., 0., 1.]])
        pred.select_hits(faces, weights)
        pred.begin(live.state, starts)
        observed = pred(np.zeros((2, 3)), np.zeros((2, 3)))
        tris = pred.runtime.model.tri_indices.numpy()[faces]
        p = np.einsum('ij,ijk->ik', weights, pred.final_points()[tris])
        expected = cap.distances(p, .008)[0]-cfg.margin
        np.testing.assert_allclose(observed[-1], expected, atol=3e-7)
        self.assertEqual(observed.shape, (3, 2))

    def test_prediction_isolation_repeatability_and_contact(self):
        model, live, pred, ids, starts, _, _ = self.fixture(contact=True)
        saved_q, saved_qd = live.state.particle_q.numpy(), live.state.particle_qd.numpy()
        work = live.solver.dx.numpy().copy()
        contacts = live.contacts.soft_contact_count.numpy().copy()
        pipeline = model._collision_pipeline
        arrays = {k: v.numpy().copy() for k, v in vars(model).items() if isinstance(v, wp.array)}
        pred.begin(live.state, starts)
        u, leaders = np.zeros((2, 3)), np.full((2, 3), .01)
        first = pred(u, leaders)
        pred(u+.03, leaders)
        repeat = pred(u, leaders)
        np.testing.assert_allclose(repeat, first, atol=1e-7)
        np.testing.assert_array_equal(saved_q, live.state.particle_q.numpy())
        np.testing.assert_array_equal(saved_qd, live.state.particle_qd.numpy())
        np.testing.assert_array_equal(work, live.solver.dx.numpy())
        np.testing.assert_array_equal(contacts, live.contacts.soft_contact_count.numpy())
        self.assertIs(pipeline, model._collision_pipeline)
        for key, values in arrays.items():
            np.testing.assert_array_equal(values, getattr(model, key).numpy(), err_msg=key)
        self.assertGreater(pred.runtime.contacts.soft_contact_count.numpy()[0], 0)

    def test_six_direction_sensitivities_against_central_and_heldout(self):
        from capsule_cbf.predictive import six_direction_sensitivity
        _, live, pred, ids, starts, _, _ = self.fixture()
        pred.begin(live.state, starts)
        u, leaders = np.zeros((2, 3)), np.zeros((2, 3))
        base = pred(u, leaders)
        jac = six_direction_sensitivity(pred, u, leaders, .02, base)
        central = []
        for d in np.eye(6):
            plus, minus = u+.01*d.reshape(2, 3), u-.01*d.reshape(2, 3)
            central.append((pred(plus, leaders)-pred(minus, leaders))/.02)
        central = np.stack(central, axis=-1)
        np.testing.assert_allclose(jac, central, atol=.006, rtol=.15)
        direction = np.array([.7, -.3, .5, -.2, .6, -.4])
        actual = pred(u+.01*direction.reshape(2, 3), leaders)
        approximate = base+.01*(jac@direction)
        self.assertLess(np.max(abs(actual-approximate)), 8e-5)
        print('Newton sensitivity max central error:', float(abs(jac-central).max()),
              'held-out barrier error:', float(abs(actual-approximate).max()))

    def test_ignored_material_fields_have_no_effect(self):
        model, live, pred, ids, starts, _, _ = self.fixture(contact=True)
        pred.begin(live.state, starts)
        u = np.zeros((2, 3))
        before = pred(u, u)
        # These buffers are stored on Model, but not read by this Style3D path.
        materials = model.tri_materials.numpy()
        materials[:, 1:3] *= 1000.
        model.tri_materials.assign(materials)
        edges = model.edge_bending_properties.numpy()
        edges[:, 1] = 100.
        model.edge_bending_properties.assign(edges)
        model.soft_contact_radius = .5
        after = pred(u, u)
        np.testing.assert_array_equal(before, after)

    @unittest.skipUnless(newton is not None and wp.is_cuda_available(), 'CUDA device unavailable')
    def test_cuda_graph_odd_even_and_snapshot_restore(self):
        # CPU fixtures validate dynamics; this additional test validates graph
        # buffer ownership/canonical output when run on a GPU workstation.
        from capsule_cbf.newton_runtime import ClothRuntime, NewtonPredictor
        for substeps in (3, 4):
            model, _, _, ids, starts, cfg, cap = self.fixture(substeps)
            # Build a fresh GPU model by using the shared headless fixture.
            from capsule_cbf.validate_newton import build_fixture
            model, live, pred, _, ids, starts = build_fixture(4, substeps, 'cuda:0', cfg)
            before = live.state.particle_q.numpy().copy()
            u, leaders = np.full((2, 3), .01), np.zeros((2, 3))
            pred.select_hits(np.array([0]), np.array([[.2, .3, .5]]))
            pred.begin(live.state, starts)
            observed = pred(u, leaders)
            self.assertEqual(observed.shape, (substeps, 1))
            expected = pred.final_points()
            np.testing.assert_array_equal(before, live.state.particle_q.numpy())
            live.advance(starts, np.vstack((u, leaders)))
            np.testing.assert_allclose(expected, live.state.particle_q.numpy(), atol=2e-6)


if __name__ == '__main__':
    unittest.main()

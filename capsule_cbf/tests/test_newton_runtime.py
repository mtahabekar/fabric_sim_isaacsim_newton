"""ClothRuntime: prescribed grippers, substep bookkeeping and CUDA graph replay."""
import unittest
import numpy as np
try:
    import newton
    import warp as wp
except ImportError:
    newton = None


def fixture(substeps=3, device='cpu', capture=False):
    from newton.solvers import style3d
    from capsule_cbf.newton_runtime import ClothRuntime
    with wp.ScopedDevice(device):
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
        b.add_ground_plane()
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0., 0., .16), wp.quat_identity()), hx=.22, hy=.22, hz=.16)
        style3d.add_cloth_grid(b, pos=wp.vec3(-.06, -.06, 1.05), rot=wp.quat_identity(), vel=wp.vec3(),
                               dim_x=4, dim_y=4, cell_x=.03, cell_y=.03, mass=.003,
                               tri_aniso_ke=wp.vec3(3.e4, 3.e4, 6.e3), edge_aniso_ke=wp.vec3(.002, .001, .0005),
                               particle_radius=.008)
        rest = np.asarray(b.particle_q, dtype=np.float32)
        wanted = np.array([[-.06, -.06], [.06, -.06], [-.06, .06], [.06, .06]])
        corners = np.array([np.argmin(np.linalg.norm(rest[:, :2]-p, axis=1)) for p in wanted])
        for i in corners:
            b.particle_flags[int(i)] &= ~int(newton.ParticleFlags.ACTIVE)
        model = b.finalize(device=device)
        model.set_gravity((0, 0, -9.81))
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 4000., .01, 2.
        model.soft_contact_margin = .012
        runtime = ClothRuntime(model, corners, np.arange(4), np.zeros((4, 3)), substeps, 1/60/substeps, 5, 8,
                               capture=capture)
    return model, runtime, corners, rest[corners].astype(float)


COMMANDS = np.array([[.12, -.05, .02], [-.07, .03, -.02], [.08, .01, .04], [.06, 0., .03]])


@unittest.skipIf(newton is None, 'Newton/Warp unavailable')
class ClothRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_grippers_follow_commands_for_odd_and_even_substeps(self):
        for substeps in (3, 4):
            model, runtime, corners, starts = fixture(substeps)
            runtime.advance(starts, COMMANDS)
            q, qd = runtime.state.particle_q.numpy(), runtime.state.particle_qd.numpy()
            np.testing.assert_allclose(q[corners], starts+COMMANDS/60, atol=1e-6)
            np.testing.assert_allclose(qd[corners], COMMANDS, atol=1e-6)
            self.assertTrue(np.isfinite(q).all())
            free = np.setdiff1d(np.arange(model.particle_count), corners)
            self.assertLess(q[free, 2].mean(), model.particle_q.numpy()[free, 2].mean())  # gravity acts

    def test_frames_compose_and_inputs_are_validated(self):
        _, runtime, corners, starts = fixture()
        for k in range(3):
            runtime.advance(starts+k*COMMANDS/60, COMMANDS)
        np.testing.assert_allclose(runtime.state.particle_q.numpy()[corners], starts+3*COMMANDS/60, atol=1e-6)
        with self.assertRaises(ValueError):
            runtime.advance(starts[:3], COMMANDS)
        with self.assertRaises(ValueError):
            runtime.advance(starts, np.full((4, 3), np.nan))

    @unittest.skipUnless(newton is not None and wp.is_cuda_available(), 'CUDA device unavailable')
    def test_cuda_graph_replay_matches_direct_stepping(self):
        for substeps in (3, 4):
            _, graph, _, starts = fixture(substeps, 'cuda:0', capture=True)
            _, direct, _, _ = fixture(substeps, 'cuda:0', capture=False)
            self.assertIsNotNone(graph.graph)
            for k in range(3):
                graph.advance(starts+k*COMMANDS/60, COMMANDS)
                direct.advance(starts+k*COMMANDS/60, COMMANDS)
            np.testing.assert_allclose(graph.state.particle_q.numpy(), direct.state.particle_q.numpy(), atol=2e-6)


if __name__ == '__main__':
    unittest.main()

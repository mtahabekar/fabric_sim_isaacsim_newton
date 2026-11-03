import unittest
import numpy as np
from capsule_cbf.cloth_model import ClothMechanics, HorizonModel

try:
    import warp as wp
    import newton
    from newton.solvers import style3d
    from newton._src.solvers.style3d.kernels import eval_stretch_kernel, eval_bend_kernel
except ImportError:  # pure NumPy environments
    newton = None


def grid_mechanics(n=4, size=.3, pinned_corners=True):
    """Small anisotropic grid with arbitrary but fixed bending weights."""
    x, y = np.meshgrid(np.linspace(0, size, n+1), np.linspace(0, size, n+1))
    rest = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    tris = []
    for j in range(n):
        for i in range(n):
            a = j*(n+1)+i
            tris += [(a, a+1, a+n+1), (a+1, a+n+2, a+n+1)]
    tris = np.array(tris)
    edges = rest[tris][:, 1:, :2]-rest[tris][:, :1, :2]
    inv_dm = np.linalg.inv(edges.transpose(0, 2, 1))  # columns are rest edges
    areas = .5*abs(np.linalg.det(edges))
    bend_edges = np.array([[0, n+2, 1, n+1], [n+2, 2*n+3, n+3, 2*n+2]])
    weights = np.array([[1., -.5, -.25, -.25], [.3, -.1, -.1, -.1]])*1e-2
    mass = np.full(len(rest), .002)
    pinned = np.zeros(len(rest), dtype=bool)
    if pinned_corners:
        pinned[[0, n, n*(n+1), (n+1)**2-1]] = True
    m = ClothMechanics(tris, inv_dm, areas, np.tile([3e3, 2e3, 6e2], (len(tris), 1)),
                       bend_edges, weights, mass, pinned, [0, 0, -9.81])
    return m, rest


class ClothMechanicsTest(unittest.TestCase):
    def setUp(self):
        self.m, self.rest = grid_mechanics()
        rng = np.random.default_rng(3)
        self.q = self.rest+rng.normal(scale=.01, size=self.rest.shape)

    def test_rest_state_is_force_free_without_bending_curvature(self):
        self.assertLess(abs(self.m.corner_forces(self.rest)).max(), 1e-9)

    def test_stiffness_matches_force_finite_differences(self):
        k = self.m.stiffness(self.q, project=False).toarray()
        eps = 1e-6
        for dof in (0, 7, 20, 41):
            d = np.zeros(self.q.size)
            d[dof] = eps
            central = (self.m.forces(self.q+d.reshape(-1, 3))-self.m.forces(self.q-d.reshape(-1, 3)))/(2*eps)
            np.testing.assert_allclose(-k[:, dof], central.ravel(), atol=1e-4*abs(k).max())

    def test_projected_stiffness_is_symmetric_psd(self):
        k = self.m.stiffness(self.q*np.array([.97, .97, 1.])).toarray()  # compressed: indefinite raw
        np.testing.assert_allclose(k, k.T, atol=1e-9*abs(k).max())
        self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-8*abs(k).max())

    def _sensitivity_error(self, q, substeps, frames, column, jacobian='trajectory'):
        nodes = np.flatnonzero(self.m.pinned)
        model = HorizonModel(self.m, nodes, np.arange(4), np.zeros((4, 3)), 1/60, substeps=substeps,
                             jacobian=jacobian)
        model.linearize(q)
        velocity, eps = np.zeros((4, 3)), 1e-6
        base = model.predict(q, np.zeros_like(q), q[nodes], velocity, frames)
        dv = np.zeros(12)
        dv[column] = eps
        plus = model.predict(q, np.zeros_like(q), q[nodes], velocity+dv.reshape(4, 3), frames)
        minus = model.predict(q, np.zeros_like(q), q[nodes], velocity-dv.reshape(4, 3), frames)
        fd = (plus['q']-minus['q'])/(2*eps)
        return [abs(base['sq'][k, ..., column]-fd[k]).max()/abs(fd[k]).max() for k in range(frames)]

    def test_frozen_sensitivity_is_exact_at_linearization_point(self):
        for column in (0, 4, 8):
            self.assertLess(self._sensitivity_error(self.q, 1, 1, column, 'frozen')[0], 1e-7)

    def test_trajectory_sensitivity_is_exact_derivative_of_scheme(self):
        # Forward sensitivity analysis: exact over every frame and substep,
        # including the slack, strongly nonlinear flat state.
        q = self.rest+np.random.default_rng(1).normal(scale=1e-4, size=self.rest.shape)
        for column in (0, 4, 8):
            self.assertLess(max(self._sensitivity_error(q, 2, 4, column)), 1e-4)

    def test_affine_sensitivity_is_exact_and_stable_under_compression(self):
        self.assertLess(max(self._sensitivity_error(self.q, 2, 4, 3, 'affine')), 1e-6)
        compressed = self.rest*np.array([.9, .9, 1.])+np.array([.015, .015, 0.])
        nodes = np.flatnonzero(self.m.pinned)
        drift = {}
        for mode in ('affine', 'trajectory'):
            model = HorizonModel(self.m, nodes, np.arange(4), np.zeros((4, 3)), 1/60, jacobian=mode)
            model.linearize(compressed)
            out = model.predict(compressed, np.zeros_like(compressed), compressed[nodes],
                                np.zeros((4, 3)), 20)
            drift[mode] = abs(out['q'][-1]-compressed).max()
        # A slack sheet linearized at slackness has no supporting stiffness:
        # bounded by free fall, whereas nonlinear modes grow exponentially.
        self.assertLess(drift['affine'], .5*9.81*(20/60)**2+.03)
        self.assertGreater(drift['trajectory'], 10*drift['affine'])

    def test_frozen_jacobian_is_wrong_for_slack_cloth(self):
        # Documents why the trajectory mode is the default.
        q = self.rest+np.random.default_rng(1).normal(scale=1e-4, size=self.rest.shape)
        self.assertGreater(max(self._sensitivity_error(q, 2, 4, 0, 'frozen')), .1)

    def test_fixed_pattern_assembly_and_banded_solve(self):
        nodes = np.flatnonzero(self.m.pinned)
        model = HorizonModel(self.m, nodes, np.arange(4), np.zeros((4, 3)), 1/60)
        model.linearize(self.q)
        k = self.m.stiffness(self.q)
        np.testing.assert_allclose(model.k_ff.toarray(), k[model.free_dofs][:, model.free_dofs].toarray(),
                                   atol=1e-9*abs(k).max())
        np.testing.assert_allclose(model.k_fp.toarray(), k[model.free_dofs][:, model.pin_dofs].toarray(),
                                   atol=1e-9*abs(k).max())
        a = np.diag(model.mass_free)+model.h**2*model.k_ff.toarray()
        b = np.random.default_rng(2).normal(size=(len(a), 3))
        np.testing.assert_allclose(a@model.solve(b), b, atol=1e-9)

    def test_gripper_resultant_and_jacobian(self):
        nodes = np.array([0, 1, 5])
        np.testing.assert_allclose(self.m.resultant(self.q, nodes), self.m.forces(self.q)[nodes].sum(axis=0))
        jac = self.m.resultant_jacobian(self.q, nodes)
        eps = 1e-6
        for node, axis in ((0, 2), (6, 0), (11, 1)):
            d = np.zeros_like(self.q)
            d[node, axis] = eps
            fd = (self.m.resultant(self.q+d, nodes)-self.m.resultant(self.q-d, nodes))/(2*eps)
            np.testing.assert_allclose(jac[:, node, axis], fd, atol=1e-5*abs(jac).max())

    def test_ks_strain_aggregate_gradient(self):
        from capsule_cbf.strain import TriangleStrain
        strain = TriangleStrain(self.m.triangles, self.m.inv_dm, .02, .03, .2)
        subset = np.arange(0, len(self.m.triangles), 2)
        value, grad = strain.aggregate(self.q, subset, kappa=30.)
        g = (strain.measures(self.q)[subset]/strain.limits).max()
        self.assertGreaterEqual(value, g)
        self.assertLessEqual(value, g+np.log(4*len(subset))/30.+1e-12)
        eps = 1e-7
        for node, axis in ((1, 0), (7, 2), (12, 1)):
            d = np.zeros_like(self.q)
            d[node, axis] = eps
            fd = (strain.aggregate(self.q+d, subset, 30.)[0]-strain.aggregate(self.q-d, subset, 30.)[0])/(2*eps)
            self.assertAlmostEqual(grad[node, axis], fd, delta=1e-4*abs(grad).max())

    def test_pinned_rows_follow_prescribed_motion(self):
        nodes = np.flatnonzero(self.m.pinned)
        model = HorizonModel(self.m, nodes, np.arange(4), np.zeros((4, 3)), 1/60)
        model.linearize(self.q)
        velocity = np.arange(12.).reshape(4, 3)/100
        out = model.predict(self.q, np.zeros_like(self.q), self.q[nodes], velocity, 2)
        np.testing.assert_allclose(out['q'][1, nodes], self.q[nodes]+2/60*velocity)
        np.testing.assert_allclose(out['sq'][1, nodes[1], :, 3:6], 2/60*np.eye(3))


@unittest.skipIf(newton is None, 'Newton/Warp unavailable')
class NewtonForceAgreementTest(unittest.TestCase):
    def test_forces_match_installed_style3d_kernels(self):
        wp.init()
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
        style3d.add_cloth_grid(b, pos=wp.vec3(), rot=wp.quat_identity(), vel=wp.vec3(), dim_x=6, dim_y=5,
                               cell_x=.04, cell_y=.05, mass=1e-3, tri_aniso_ke=wp.vec3(3e4, 2e4, 6e3),
                               edge_aniso_ke=wp.vec3(.002, .001, .0005), particle_radius=.008)
        model = b.finalize(device='cpu')
        rng = np.random.default_rng(0)
        q = model.particle_q.numpy()+rng.normal(scale=.004, size=(model.particle_count, 3)).astype(np.float32)
        qw = wp.array(q, dtype=wp.vec3, device='cpu')
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device='cpu')
        wp.launch(eval_stretch_kernel, model.tri_count, inputs=[qw, model.tri_areas, model.tri_poses,
                  model.tri_indices, model.style3d.tri_aniso_ke], outputs=[forces], device='cpu')
        wp.launch(eval_bend_kernel, model.edge_count, inputs=[qw, model.style3d.edge_rest_area,
                  model.style3d.edge_bending_cot, model.edge_indices, model.edge_bending_properties],
                  outputs=[forces], device='cpu')
        mechanics = ClothMechanics.from_newton(model)
        mine = mechanics.forces(q.astype(float))
        np.testing.assert_allclose(mine, forces.numpy(), atol=2e-5*abs(mine).max(), rtol=0)


if __name__ == '__main__':
    unittest.main()

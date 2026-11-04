import unittest
import numpy as np
from capsule_cbf.strain import TriangleStrain


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

    def test_local_extension_is_measured_per_triangle(self):
        points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [.5, .5, 1.]], float)
        rest = np.column_stack((points[1, :2]-points[0, :2], points[4, :2]-points[0, :2]))
        strain = TriangleStrain([[0, 1, 4]], [np.linalg.inv(rest)])
        points[4, 2] += .2  # interior lift with the cloth corners unchanged
        self.assertLess(strain.barriers(points).min(), 0)

    def test_batched_ks_aggregate_matches_single_evaluations(self):
        rng = np.random.default_rng(5)
        tris = np.array([[0, 1, 2], [1, 3, 2]])
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
        inv = np.linalg.inv(np.stack([np.column_stack((rest[t[1], :2]-rest[t[0], :2], rest[t[2], :2]-rest[t[0], :2]))
                                      for t in tris]))
        strain = TriangleStrain(tris, inv)
        batch = rest+rng.normal(scale=.01, size=(4, 4, 3))
        values, grads = strain.aggregate(batch, np.arange(2))
        for k in range(4):
            value, grad = strain.aggregate(batch[k], np.arange(2))
            self.assertAlmostEqual(values[k], value)
            np.testing.assert_allclose(grads[k], grad)

    def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            TriangleStrain([[0, 1, 2]], [np.eye(2)], shear_limit=1.)


if __name__ == '__main__':
    unittest.main()

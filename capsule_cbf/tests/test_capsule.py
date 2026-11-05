import itertools
import unittest
import numpy as np
from capsule_cbf.capsule import Capsule

BOX_CORNERS = np.array(list(itertools.product((-.22, .22), (-.22, .22), (0., .32))))


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.hemisphere = Capsule([0., 0., .16], [.22, .22, .16], top_cap_mode='hemisphere')
        self.sphere = Capsule([0., 0., .16], [.22, .22, .16])  # default: scale 2
        self.centered = Capsule([0., 0., .16], [.22, .22, .16], top_radius_scale=1.)
        self.caps = (self.hemisphere, self.sphere, self.centered)

    def surface_samples(self, cap):
        return cap.surface(np.linspace(0, 1, 37), np.linspace(0, 2*np.pi, 24, endpoint=False))

    def test_box_enclosure_and_surface(self):
        for cap in self.caps:
            self.assertTrue(np.all(cap.distances(BOX_CORNERS)[0] <= 1e-12))
            np.testing.assert_allclose(cap.distances(self.surface_samples(cap))[0], 0., atol=1e-12)
            self.assertLess(cap.distances([[0., 0., .16]])[0][0], 0.)

    def test_centered_sphere_scale_one_is_box_centered(self):
        radius = np.sqrt(2*.22**2+.16**2)
        self.assertAlmostEqual(self.centered.top_sphere_radius, radius)
        self.assertAlmostEqual(self.centered.top_center, 0.)
        peak = self.centered.surface([1.], [0.])[0]
        np.testing.assert_allclose(peak, [0, 0, .16+radius], atol=1e-12)
        samples = self.surface_samples(self.centered)
        top = samples[samples[:, 2] > .32+1e-9]
        np.testing.assert_allclose(np.linalg.norm(top-self.centered.center, axis=1), radius)

    def test_default_cap_doubles_radius_through_top_corners(self):
        radius = 2*np.sqrt(2*.22**2+.16**2)
        cap = self.sphere
        self.assertEqual(cap.top_cap_mode, 'centered_sphere')
        self.assertAlmostEqual(cap.top_sphere_radius, radius)
        sphere_center = cap.center+[0, 0, cap.top_center]
        top_corners = BOX_CORNERS[BOX_CORNERS[:, 2] > .3]
        np.testing.assert_allclose(np.linalg.norm(top_corners-sphere_center, axis=1), radius, atol=1e-12)
        np.testing.assert_allclose(cap.distances(top_corners)[0], 0., atol=1e-12)
        peak = cap.surface([1.], [0.])[0]
        self.assertAlmostEqual(peak[2]-.32, radius-np.sqrt(radius**2-2*.22**2))  # 7.3 cm
        self.assertLess(peak[2], self.centered.surface([1.], [0.])[0, 2])
        self.assertLess(self.centered.surface([1.], [0.])[0, 2], self.hemisphere.surface([1.], [0.])[0, 2])
        samples = self.surface_samples(cap)
        top = samples[samples[:, 2] > .32+1e-9]
        np.testing.assert_allclose(np.linalg.norm(top-sphere_center, axis=1), radius)

    def test_signed_distance_normals_and_projection(self):
        for scale in (1., 2., 3.5):
            cap = Capsule([.1, -.2, .16], [.22, .22, .16], top_radius_scale=scale)
            points = np.random.default_rng(7).uniform(-.7, .7, (300, 3))+cap.center
            d, normals, surface = cap.distances(points)
            np.testing.assert_allclose(cap.distances(surface)[0], 0., atol=1e-12)
            np.testing.assert_allclose(points-surface, d[:, None]*normals, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-12)
            np.testing.assert_allclose(cap.distances(points, .008)[0], d-.008)
            # Brute force: distance to a dense surface sample, sign by containment.
            dense = cap.surface(np.linspace(0, 1, 1500), np.linspace(0, 2*np.pi, 360, endpoint=False))
            near = np.min(np.linalg.norm(points[:40, None]-dense[None], axis=2), axis=1)
            np.testing.assert_allclose(np.abs(d[:40]), near, atol=2e-3)

    def test_normals_are_distance_gradients(self):
        rng = np.random.default_rng(3)
        for cap in self.caps:
            points = rng.uniform(-.7, .7, (200, 3))+cap.center
            d, normals, _ = cap.distances(points, .008)
            for axis in range(3):
                step = np.eye(3)[axis]*1e-6
                numerical = (cap.distances(points+step, .008)[0]-cap.distances(points-step, .008)[0])/2e-6
                np.testing.assert_allclose(numerical, normals[:, axis], atol=1e-6)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            Capsule([0, 0, 0], [.2, 0., .1])
        with self.assertRaises(ValueError):
            Capsule([0, 0, 0], [.2, .2, .1], top_cap_mode='cone')
        with self.assertRaises(ValueError):  # sphere narrower than the top-face diagonal
            Capsule([0, 0, 0], [.2, .2, .1], top_radius_scale=.5)


if __name__ == '__main__':
    unittest.main()

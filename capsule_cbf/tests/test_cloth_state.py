import unittest
from types import SimpleNamespace
import numpy as np
from capsule_cbf.cloth_state import ClothState, particle_state_payload, unpack_cloth_state
from capsule_cbf.core import Capsule


class ClothStateTests(unittest.TestCase):
    def test_snapshot_slice_copies_all_particle_velocities(self):
        q = np.arange(21, dtype=float).reshape(7, 3)
        v = -q.copy()
        source = SimpleNamespace(particle_q=SimpleNamespace(numpy=lambda: q),
                                 particle_qd=SimpleNamespace(numpy=lambda: v))
        state = ClothState.from_newton(source, 1, 5)
        np.testing.assert_array_equal(state.velocities, -q[1:6])
        self.assertEqual(state.particle_start, 1)
        q[:] = v[:] = 0
        self.assertEqual(state.positions[2, 0], 9)
        self.assertEqual(state.velocities[2, 0], -9)
        with self.assertRaises(ValueError):
            state.velocities[0, 0] = 1
        with self.assertRaises(ValueError):
            ClothState.from_newton(source, 5, 3)

    def test_clearance_rates_match_motion_for_every_particle(self):
        cap = Capsule([0, 0, 0], [.1, .1, .2])
        q = np.array([[.3, 0, 0], [0, .3, 0], [0, 0, .5], [.2, .2, .4], [-.3, 0, 0]])
        v = np.array([[-.1, 0, 0], [0, .2, 0], [0, 0, -.3], [.1, -.2, .1], [-.2, 0, 0]])
        state = ClothState(q, v)
        central = (cap.distances(q+1e-6*v)[0]-cap.distances(q-1e-6*v)[0])/2e-6
        np.testing.assert_allclose(state.clearance_rates(cap), central, atol=1e-9)
        points, debug = unpack_cloth_state(state, cap)
        self.assertIs(points, state.positions)
        self.assertEqual(debug['state_particle_count'], 5)
        self.assertEqual(debug['approaching_particle_count'], 2)
        self.assertFalse(unpack_cloth_state(q, cap)[1]['state_velocity_available'])

    def test_invalid_state_is_rejected(self):
        for q, v in ((np.zeros((2, 3)), np.zeros((1, 3))),
                     (np.zeros((2, 3)), np.full((2, 3), np.nan))):
            with self.assertRaises(ValueError):
                ClothState(q, v)

    def test_ros_payload_has_xyz_and_velocity_fields_in_particle_order(self):
        q, v = np.arange(15).reshape(5, 3), -np.arange(15).reshape(5, 3)
        names, packed = particle_state_payload(q, v)
        self.assertEqual(names, ('x', 'y', 'z', 'vx', 'vy', 'vz'))
        self.assertEqual(packed.strides, (24, 4))
        decoded = np.frombuffer(packed.tobytes(), dtype=np.dtype([(n, '<f4') for n in names]))
        np.testing.assert_array_equal(decoded['z'], q[:, 2])
        np.testing.assert_array_equal(decoded['vz'], v[:, 2])
        old_names, old = particle_state_payload(q)
        self.assertEqual(old_names, ('x', 'y', 'z'))
        self.assertEqual(old.strides, (12, 4))


if __name__ == '__main__':
    unittest.main()

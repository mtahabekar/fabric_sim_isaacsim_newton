"""Coarse material grid: node selection, barycentric lifting, coarse model mass and obstacles."""
import unittest
import numpy as np
try:
    import newton
    import warp as wp
except ImportError:
    newton = None


def square_patches(rest, half_width):
    patch, ids = [], []
    lo, hi = rest[:, :2].min(axis=0), rest[:, :2].max(axis=0)
    for cid, tip in enumerate(((lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1]))):
        members = np.flatnonzero(np.all(abs(rest[:, :2]-tip) <= half_width, axis=1))
        patch.extend(members)
        ids.extend([cid]*len(members))
    return np.array(patch), np.array(ids)


@unittest.skipIf(newton is None, 'Newton/Warp unavailable')
class MaterialGridTests(unittest.TestCase):
    def test_patch_boundaries_are_inserted_and_points_lift_exactly(self):
        from capsule_cbf.material_grid import MaterialGrid
        rest = np.array([[x, y, .5] for y in range(9) for x in range(9)], dtype=float)
        patch, ids = square_patches(rest, 1)
        grid = MaterialGrid(rest, patch, ids, 2)
        # A coarse base resolution still keeps every gripper patch boundary line.
        np.testing.assert_array_equal(grid.x, [0, 1, 4, 7, 8])
        np.testing.assert_array_equal(grid.y, grid.x)
        np.testing.assert_array_equal(rest[grid.source], grid.vertices)
        uv = np.random.default_rng(19).uniform(0, 8, (50, 2))
        faces, w = grid.locate(uv)
        mapped = np.einsum('ij,ijk->ik', w, grid.vertices[grid.triangles[faces]])
        np.testing.assert_allclose(mapped[:, :2], uv, atol=1e-5)
        with self.assertRaises(ValueError):
            grid.locate([[9., 0.]])

    def test_non_rectangular_or_bent_cloth_rejected(self):
        from capsule_cbf.material_grid import MaterialGrid
        rest = np.array([[x, y, .5] for y in range(5) for x in range(5)], dtype=float)
        patch, ids = square_patches(rest, 0)
        bent = rest.copy()
        bent[3, 2] += .1
        with self.assertRaises(ValueError):
            MaterialGrid(bent, patch, ids, 2)
        with self.assertRaises(ValueError):
            MaterialGrid(rest[:-1], patch[:-1], ids[:-1], 2)

    def test_coarse_model_preserves_mass_materials_obstacles_and_pins(self):
        from newton.solvers import style3d
        from capsule_cbf.material_grid import MaterialGrid, build_coarse_model
        wp.init()
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
        b.add_ground_plane()
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0., 0., .16), wp.quat_identity()), hx=.22, hy=.22, hz=.16)
        start = len(b.particle_q)
        style3d.add_cloth_grid(b, pos=wp.vec3(-.12, -.12, 1.), rot=wp.quat_identity(), vel=wp.vec3(), dim_x=8, dim_y=8,
                               cell_x=.03, cell_y=.03, mass=.001, tri_aniso_ke=wp.vec3(3.e4, 3.e4, 6.e3),
                               edge_aniso_ke=wp.vec3(.002, .001, .0005), particle_radius=.008)
        rest = np.asarray(b.particle_q[start:], dtype=float)
        local, ids = square_patches(rest, .031)
        offsets = rest[local]-rest[[local[ids == c][0] for c in ids]]
        for i in local:
            b.particle_flags[start+int(i)] &= ~int(newton.ParticleFlags.ACTIVE)
        model = b.finalize(device='cpu')
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 4000., .01, 2.
        model.soft_contact_margin = .012
        grid = MaterialGrid(rest, local, ids, 3)
        coarse, pins, pin_ids, _ = build_coarse_model(model, grid, local+start, ids, offsets, start, .008)
        self.assertLess(coarse.particle_count, model.particle_count-start)
        self.assertAlmostEqual(float(coarse.particle_mass.numpy().sum()),
                               float(model.particle_mass.numpy()[start:].sum()), places=6)
        np.testing.assert_allclose(coarse.style3d.tri_aniso_ke.numpy()[0], [3.e4, 3.e4, 6.e3])
        for name in ('shape_type', 'shape_transform', 'shape_scale'):
            np.testing.assert_array_equal(getattr(model, name).numpy(), getattr(coarse, name).numpy())
        flags = coarse.particle_flags.numpy()
        np.testing.assert_array_equal(np.sort(np.flatnonzero((flags & 1) == 0)), np.sort(pins))
        self.assertEqual(set(pin_ids.tolist()), {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()

"""Coarse material grid of a rectangular cloth and its Newton model.

The coarse nodes are a subset of the fine grid in material (rest XY)
coordinates, always including the gripper patch boundary lines; fine points
map to coarse triangles by barycentric weights. build_coarse_model creates a
Style3D cloth on those nodes with the same areal density, materials and static
obstacles, from which ReducedCloth copies the mechanics. Supported: one flat
rectangular grid, uniform material and density, four gripper patches.
"""
import numpy as np
import newton
import warp as wp
from newton.solvers import style3d


def grid_triangles(nx, ny):
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            b, c, d = a+1, a+nx+1, a+nx+2
            faces.extend(((a, b, c), (b, d, c)))
    return np.asarray(faces, dtype=np.int32)


class MaterialGrid:
    """Fixed mappings in rest/material coordinates, never nearest deformed XYZ."""
    def __init__(self, rest, patch_indices, corner_ids, cells):
        self.rest = np.asarray(rest)
        if not np.allclose(self.rest[:, 2], self.rest[0, 2], atol=1e-6):
            raise ValueError('Coarse prediction requires an initially flat XY cloth')
        x, y = [np.unique(np.round(self.rest[:, axis], 6)) for axis in (0, 1)]
        if len(x)*len(y) != len(rest) or min(len(x), len(y)) < 3:
            raise ValueError('Coarse prediction requires a rectangular material grid')
        ix = np.abs(self.rest[:, 0, None]-x).argmin(axis=1)
        iy = np.abs(self.rest[:, 1, None]-y).argmin(axis=1)
        flat = iy*len(x)+ix
        if len(np.unique(flat)) != len(rest):
            raise ValueError('Duplicate material grid coordinates')
        lookup = np.empty((len(y), len(x)), dtype=int)
        lookup[iy, ix] = np.arange(len(rest))
        axes = []
        for coord, indices in ((x, ix), (y, iy)):
            chosen = np.rint(np.linspace(0, len(coord)-1, min(cells, len(coord)-1)+1)).astype(int)
            # Insert patch boundary lines without retaining its interior grid.
            boundary = []
            for corner in range(4):
                values = indices[patch_indices[np.asarray(corner_ids) == corner]]
                if not len(values):
                    raise ValueError('All four gripper patches are required')
                boundary.extend((values.min(), values.max()))
            chosen = np.unique(np.concatenate((chosen, boundary)))
            axes.append(chosen)
        self.source = lookup[np.ix_(axes[1], axes[0])].ravel().astype(np.int32)
        self.vertices = self.rest[self.source].copy()
        # Use exact builder coordinates (rounding above was for identification).
        self.x = self.vertices[:len(axes[0]), 0]
        self.y = self.vertices[::len(axes[0]), 1]
        self.triangles = grid_triangles(len(self.x)-1, len(self.y)-1)

    def locate(self, uv):
        uv = np.asarray(uv).reshape(-1, 2)
        if (not np.isfinite(uv).all() or np.any(uv < [self.x[0]-1e-6, self.y[0]-1e-6])
                or np.any(uv > [self.x[-1]+1e-6, self.y[-1]+1e-6])):
            raise ValueError('Material point outside cloth')
        i = np.clip(np.searchsorted(self.x, uv[:, 0], side='right')-1, 0, len(self.x)-2)
        j = np.clip(np.searchsorted(self.y, uv[:, 1], side='right')-1, 0, len(self.y)-2)
        a = np.clip((uv[:, 0]-self.x[i])/(self.x[i+1]-self.x[i]), 0., 1.)
        b = np.clip((uv[:, 1]-self.y[j])/(self.y[j+1]-self.y[j]), 0., 1.)
        upper = a+b > 1.
        faces = 2*(j*(len(self.x)-1)+i)+upper
        weights = np.where(upper[:, None], np.column_stack((1-b, a+b-1, 1-a)),
                           np.column_stack((1-a-b, a, b)))
        return faces.astype(np.int32), weights.astype(np.float32)


def uniform(array, label):
    values = array.numpy()
    if not len(values) or not np.allclose(values, values[0], rtol=1e-5, atol=1e-8):
        raise ValueError('Coarse prediction requires uniform '+label)
    return values[0]


def build_coarse_model(model, grid, patch, corner_ids, offsets, particle_start, particle_radius):
    if model.body_count or model.world_count != 1:
        raise ValueError('Coarse prediction requires one world and static obstacles')
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(b)
    shape_fields = ('type', 'transform', 'scale', 'flags', 'margin', 'gap', 'is_solid', 'collision_group')
    shapes = {key: getattr(model, 'shape_'+key).numpy() for key in shape_fields}
    materials = {key: getattr(model, 'shape_material_'+key).numpy()
                 for key in ('ke', 'kd', 'kf', 'ka', 'mu', 'restitution', 'mu_torsional', 'mu_rolling', 'kh')}
    for i in range(model.shape_count):
        if shapes['type'][i] not in (int(newton.GeoType.PLANE), int(newton.GeoType.BOX)):
            raise ValueError('Coarse prediction currently supports static planes and boxes')
        cfg = newton.ModelBuilder.ShapeConfig(density=0., **{k: float(v[i]) for k, v in materials.items()})
        cfg.flags = int(shapes['flags'][i])
        cfg.margin, cfg.gap = float(shapes['margin'][i]), float(shapes['gap'][i])
        cfg.is_solid, cfg.collision_group = bool(shapes['is_solid'][i]), int(shapes['collision_group'][i])
        transform = shapes['transform'][i]
        b.add_shape(body=-1, type=int(shapes['type'][i]),
                    xform=wp.transform(wp.vec3(*transform[:3]), wp.quat(*transform[3:])),
                    scale=wp.vec3(*shapes['scale'][i]), cfg=cfg)
    b.shape_collision_filter_pairs.extend(model.shape_collision_filter_pairs)
    masses = model.particle_mass.numpy()[particle_start:particle_start+len(grid.rest)]
    mass = masses.sum()
    area = model.tri_areas.numpy().sum()
    lumped_area = np.zeros(len(grid.rest))
    np.add.at(lumped_area, (model.tri_indices.numpy()-particle_start).ravel(),
              np.repeat(model.tri_areas.numpy()/3, 3))
    if not np.allclose(masses, lumped_area*(mass/area), rtol=2e-4, atol=1e-8):
        raise ValueError('Coarse prediction requires uniform areal density')
    # Preserve areal density, NOT mass per vertex: coarse vertices cover more cloth.
    style3d.add_cloth_mesh(b, pos=wp.vec3(), rot=wp.quat_identity(), vel=wp.vec3(),
                          vertices=grid.vertices.tolist(), indices=grid.triangles.ravel().tolist(),
                          density=float(mass/area), panel_verts=grid.vertices[:, :2].tolist(),
                          tri_aniso_ke=wp.vec3(*uniform(model.style3d.tri_aniso_ke, 'stretch material')),
                          edge_aniso_ke=wp.vec3(*uniform(model.style3d.aniso_ke, 'bending material')),
                          particle_radius=particle_radius)
    fine_to_coarse = {int(index)+particle_start: i for i, index in enumerate(grid.source)}
    coarse_patch, ids, offs = [], [], []
    for index, cid, offset in zip(patch, corner_ids, offsets):
        if int(index) in fine_to_coarse:
            k = fine_to_coarse[int(index)]
            coarse_patch.append(k)
            ids.append(cid)
            offs.append(offset)
            b.particle_flags[k] &= ~int(newton.ParticleFlags.ACTIVE)
    coarse = b.finalize(device=model.device)
    coarse.set_gravity(model.gravity.numpy())
    for name in ('soft_contact_ke', 'soft_contact_kd', 'soft_contact_mu', 'soft_contact_margin'):
        setattr(coarse, name, getattr(model, name))
    return coarse, np.asarray(coarse_patch), np.asarray(ids), np.asarray(offs)

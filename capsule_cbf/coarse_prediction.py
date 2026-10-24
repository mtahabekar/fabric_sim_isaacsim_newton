"""Coarse dynamics with fine-mesh sensing and frozen subgrid displacement.

Supported material domain: one flat XY rectangular grid, uniform Style3D
material/density, four rectangular gripper patches, static plane/box obstacles.
State restriction and geometric residuals run on the model's Warp device.
"""
import numpy as np
import newton
import warp as wp
from newton.solvers import style3d
from .newton_runtime import NewtonPredictor
from .cloth_state import ClothState


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


@wp.kernel
def restrict_state(source: wp.array[int], fine_q: wp.array[wp.vec3], fine_v: wp.array[wp.vec3],
                   q: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    i = wp.tid()
    q[i] = fine_q[source[i]]
    v[i] = fine_v[source[i]]


@wp.kernel
def hit_residual(fine_q: wp.array[wp.vec3], fine_tri: wp.array2d[int], fine_faces: wp.array[int],
                 fine_weights: wp.array[wp.vec3], coarse_q: wp.array[wp.vec3], coarse_tri: wp.array2d[int],
                 coarse_faces: wp.array[int], coarse_weights: wp.array[wp.vec3], offsets: wp.array[wp.vec3]):
    i = wp.tid()
    f, w = fine_faces[i], fine_weights[i]
    p = w[0]*fine_q[fine_tri[f, 0]]+w[1]*fine_q[fine_tri[f, 1]]+w[2]*fine_q[fine_tri[f, 2]]
    cf, cw = coarse_faces[i], coarse_weights[i]
    c = cw[0]*coarse_q[coarse_tri[cf, 0]]+cw[1]*coarse_q[coarse_tri[cf, 1]]+cw[2]*coarse_q[coarse_tri[cf, 2]]
    offsets[i] = p-c


@wp.kernel
def lift_displacement(fine_q: wp.array[wp.vec3], start: int, coarse_q: wp.array[wp.vec3],
                      coarse_initial: wp.array[wp.vec3], triangles: wp.array2d[int],
                      faces: wp.array[int], weights: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    f, w = faces[i], weights[i]
    displacement = wp.vec3(0.)
    for k in range(3):
        vertex = triangles[f, k]
        displacement += w[k]*(coarse_q[vertex]-coarse_initial[vertex])
    out[i] = fine_q[start+i]+displacement


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


class CoarseNewtonPredictor(NewtonPredictor):
    def __init__(self, model, patch_indices, corner_ids, offsets, substeps, dt,
                 iterations, linear_iterations, capsule, particle_radius, settings,
                 *, particle_start=0, particle_count=None, capture=True):
        if not settings.ray_basins_enabled or settings.strain_enabled:
            raise ValueError('Coarse predictor supports selected ray hits without strain')
        n = particle_count or model.particle_count
        rest = model.particle_q.numpy()[particle_start:particle_start+n]
        self.fine_triangles = model.tri_indices.numpy()
        if np.any(self.fine_triangles < particle_start) or np.any(self.fine_triangles >= particle_start+n):
            raise ValueError('Coarse predictor requires a single cloth triangle mesh')
        patch = np.asarray(patch_indices)
        flags = model.particle_flags.numpy()[particle_start:particle_start+n]
        if set(np.flatnonzero((flags & 1) == 0)) != set(patch-particle_start):
            raise ValueError('All pinned particles must belong to the supplied gripper patches')
        self.grid = MaterialGrid(rest, patch-particle_start, corner_ids, settings.prediction_grid_cells)
        coarse, patch, ids, offs = build_coarse_model(model, self.grid, patch, corner_ids, offsets,
                                                     particle_start, particle_radius)
        super().__init__(coarse, patch, ids, offs, substeps, dt, iterations, linear_iterations,
                         capsule, particle_radius, settings, capture=capture)
        self.fine_model = model
        self.fine_start, self.fine_count = particle_start, n
        self.fine_snapshot = model.state()
        self.restricted_state = coarse.state()
        self.source = wp.array(self.grid.source+particle_start, dtype=int, device=model.device)
        faces, weights = self.grid.locate(rest[:, :2])
        self.lift_faces = wp.array(faces, dtype=int, device=model.device)
        self.lift_weights = wp.array(weights, dtype=wp.vec3, device=model.device)
        self.lifted = wp.empty(n, dtype=wp.vec3, device=model.device)
        capacity = len(self.runtime.hit_faces)
        self.fine_hit_faces = wp.zeros(capacity, dtype=int, device=model.device)
        self.fine_hit_weights = wp.zeros(capacity, dtype=wp.vec3, device=model.device)
        self.resolution_info = dict(prediction_model='coarse', live_particles=n,
                                    prediction_particles=coarse.particle_count,
                                    prediction_triangles=coarse.tri_count,
                                    prediction_grid_shape=[len(self.grid.x)-1, len(self.grid.y)-1],
                                    prediction_geometry='fine initial position + coarse displacement')

    def begin(self, live_state, starts):
        self.fine_snapshot.assign(live_state)
        wp.launch(restrict_state, len(self.source), inputs=[self.source, live_state.particle_q,
                  live_state.particle_qd, self.restricted_state.particle_q, self.restricted_state.particle_qd],
                  device=self.runtime.device)
        super().begin(self.restricted_state, starts)
        self.selected_hit_count = None  # offsets must be refreshed for this frame

    def current_cloth_state(self):
        if self.starts is None:
            raise RuntimeError('Call begin before reading predictor state')
        return ClothState.from_newton(self.fine_snapshot, self.fine_start, self.fine_count)

    def select_hits(self, faces, weights):
        if self.starts is None:
            raise RuntimeError('Call begin before selecting coarse material hits')
        faces, weights = np.asarray(faces, dtype=int), np.asarray(weights, dtype=float)
        if (weights.shape != (len(faces), 3) or np.any(faces < 0) or np.any(faces >= len(self.fine_triangles))
                or not np.isfinite(weights).all() or np.any(weights < -1e-6)
                or not np.allclose(weights.sum(axis=1), 1., atol=1e-6)):
            raise ValueError('Invalid fine material hits')
        indices = self.fine_triangles[faces]-self.fine_start
        uv = np.einsum('ij,ijk->ik', weights, self.grid.rest[indices, :2])
        coarse_faces, coarse_weights = self.grid.locate(uv)
        super().select_hits(coarse_faces, coarse_weights)
        self.selected_fine_faces, self.selected_fine_weights = faces.copy(), weights.copy()
        capacity = len(self.fine_hit_faces)
        f, w = np.zeros(capacity, np.int32), np.zeros((capacity, 3), np.float32)
        f[:len(faces)], w[:len(faces)] = faces, weights
        self.fine_hit_faces.assign(f)
        self.fine_hit_weights.assign(w)
        r = self.runtime
        if len(faces):
            wp.launch(hit_residual, len(faces), inputs=[self.fine_snapshot.particle_q,
                      self.fine_model.tri_indices, self.fine_hit_faces, self.fine_hit_weights,
                      self.snapshot.particle_q, r.model.tri_indices, r.hit_faces, r.hit_weights,
                      r.hit_offsets], device=r.device)

    def __call__(self, followers, leaders):
        if self.selected_hit_count is None:
            raise RuntimeError('Select fine-mesh material hits before a coarse prediction')
        return super().__call__(followers, leaders)

    def final_points(self):
        r = self.runtime
        wp.launch(lift_displacement, self.fine_count, inputs=[self.fine_snapshot.particle_q,
                  self.fine_start, r.state.particle_q, self.snapshot.particle_q, r.model.tri_indices,
                  self.lift_faces, self.lift_weights, self.lifted], device=r.device)
        return self.lifted.numpy().copy()


def create_predictor(model, patch_indices, corner_ids, offsets, substeps, dt,
                     iterations, linear_iterations, capsule, particle_radius, settings, **kwargs):
    cls = CoarseNewtonPredictor if settings.prediction_grid_cells else NewtonPredictor
    return cls(model, patch_indices, corner_ids, offsets, substeps, dt, iterations,
               linear_iterations, capsule, particle_radius, settings, **kwargs)

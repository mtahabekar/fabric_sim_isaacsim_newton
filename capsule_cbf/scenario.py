"""Headless Newton plant (no Isaac, no ROS) shared by the benchmark, validation and tests.

Geometry, material and solver defaults equal the Isaac configuration
(fabric_sim_bringup/config/full_sim.yaml: 56x56 cells of 1.5 cm, 0.2 kg/m^2,
0.44x0.44x0.32 m table, 8 substeps, 10 PD and 16 PCG iterations).
Corners: 0/1 followers (y=-0.42), 2/3 leaders (y=+0.42).
"""
from dataclasses import dataclass
import numpy as np
import warp as wp
import newton
from newton.solvers import style3d
from .capsule import DEFAULT_TOP_RADIUS_SCALE, Capsule
from .newton_runtime import ClothRuntime

FRAME_DT = 1/60


@dataclass
class Plant:
    model: object
    runtime: object
    capsule: object
    corners: np.ndarray       # fine indices of the four corner particles
    patch: np.ndarray         # pinned fine indices
    corner_ids: np.ndarray
    offsets: np.ndarray
    rest: np.ndarray
    starts: np.ndarray        # current corner targets, updated by step()
    particle_radius: float = .008

    def state(self):
        return (self.runtime.state.particle_q.numpy().astype(float),
                self.runtime.state.particle_qd.numpy().astype(float))

    def step(self, followers, leaders):
        commands = np.vstack((followers, leaders)).astype(float)
        self.runtime.advance(self.starts, commands)
        self.starts = self.starts+commands*FRAME_DT

    def save(self):
        saved = self.model.state()
        saved.assign(self.runtime.state)
        return saved, self.starts.copy()

    def restore(self, snapshot):
        saved, starts = snapshot
        self.runtime.state.assign(saved)
        self.runtime.scratch.assign(saved)
        self.starts = starts.copy()


def mass_per_particle(areal_density, grid, size=.84):
    """Style3D add_cloth_grid converts per-particle mass to uniform density."""
    return areal_density*size*size/(grid+1)**2


def build_plant(device='cuda:0', grid=56, areal_density=.2, cloth_z=.85, substeps=8,
                iterations=10, linear_iterations=16, tri_ke=(3.e4, 3.e4, 6.e3),
                edge_ke=(.002, .001, .0005), capture=True, top_cap_mode='centered_sphere', inset=0.,
                top_radius_scale=DEFAULT_TOP_RADIUS_SCALE):
    """inset > 0: grippers pre-rolled that far toward the center along x and y (the
    Isaac default is 0.05 m; plant.starts then holds the inset corner positions)."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(b)
    shape = newton.ModelBuilder.ShapeConfig(density=0., mu=.8, ke=5.e5, kd=10.)
    b.add_ground_plane(cfg=shape)
    b.add_shape_box(body=-1, cfg=shape, xform=wp.transform(wp.vec3(0, 0, .16), wp.quat_identity()),
                    hx=.22, hy=.22, hz=.16)
    cell = .84/grid
    style3d.add_cloth_grid(b, pos=wp.vec3(-.42, -.42, cloth_z), rot=wp.quat_identity(), vel=wp.vec3(),
                          dim_x=grid, dim_y=grid, cell_x=cell, cell_y=cell,
                          mass=mass_per_particle(areal_density, grid),
                          tri_aniso_ke=wp.vec3(*tri_ke), tri_ka=5.e3, tri_kd=5.e-5,
                          edge_aniso_ke=wp.vec3(*edge_ke), edge_kd=1.e-5, particle_radius=.008)
    rest = np.asarray(b.particle_q, dtype=float)
    wanted = np.array([[-.42, -.42], [.42, -.42], [-.42, .42], [.42, .42]])
    corners = np.array([np.argmin(np.linalg.norm(rest[:, :2]-p, axis=1)) for p in wanted])
    patch, ids, offsets = [], [], []
    reach = min(3, max(1, grid//4))
    for cid, p in enumerate(wanted):
        for index in np.flatnonzero(np.all(abs(rest[:, :2]-p) <= (reach-1)*cell+1e-5, axis=1)):
            patch.append(index)
            ids.append(cid)
            offsets.append(rest[index]-rest[corners[cid]])
            b.particle_flags[int(index)] &= ~int(newton.ParticleFlags.ACTIVE)
    model = b.finalize(device=device)
    model.set_gravity((0, 0, -9.81))
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 4000., .01, 2.
    model.soft_contact_margin = .012
    runtime = ClothRuntime(model, patch, ids, offsets, substeps, FRAME_DT/substeps, iterations,
                           linear_iterations, capture=capture and wp.get_device(device).is_cuda)
    capsule = Capsule([0, 0, .16], [.22, .22, .16], top_cap_mode, top_radius_scale)
    starts = runtime.inset_corners(rest[corners], inset, FRAME_DT)
    return Plant(model, runtime, capsule, corners, np.asarray(patch), np.asarray(ids),
                 np.asarray(offsets, dtype=float), rest, starts)

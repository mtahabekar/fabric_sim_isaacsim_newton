"""Capsule experiment model with prescribed gripper DOFs and shared stepping."""
import numpy as np
import warp as wp
import newton
from newton.solvers import style3d
from .newton_runtime import ClothRuntime


class CurrentNewtonModelMixin:
    def _build_model(self):
        cloth_origin = wp.vec3(
            self.cloth_center_x - 0.5 * self.cloth_width,
            self.cloth_center_y - 0.5 * self.cloth_height,
            self.cloth_start_z,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=0.8,
            ke=5.0e5,
            kd=1.0e1,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=0.8,
            ke=5.0e5,
            kd=1.0e1,
        )
        builder.add_shape_box(
            body=-1,
            cfg=table_cfg,
            xform=wp.transform(p=self.table_center, q=wp.quat_identity()),
            hx=float(self.table_half_extents[0]),
            hy=float(self.table_half_extents[1]),
            hz=float(self.table_half_extents[2]),
        )

        self.cloth_particle_start = len(builder.particle_q)
        style3d.add_cloth_grid(
            builder,
            pos=cloth_origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.grid_cells_x,
            dim_y=self.grid_cells_y,
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            mass=self.args.mass_per_particle,
            tri_aniso_ke=wp.vec3(self.args.tri_ke_weft, self.args.tri_ke_warp, self.args.tri_ke_shear),
            tri_ka=self.args.tri_ka,
            tri_kd=self.args.tri_kd,
            edge_aniso_ke=wp.vec3(self.args.edge_ke_weft, self.args.edge_ke_warp, self.args.edge_ke_shear),
            edge_kd=self.args.edge_kd,
            particle_radius=self.cloth_particle_radius,
        )
        self.cloth_particle_end = len(builder.particle_q)

        builder_particle_positions = np.asarray(
            builder.particle_q[self.cloth_particle_start:self.cloth_particle_end],
            dtype=np.float32,
        )

        (
            self.corner_indices,
            self.gripper_flat_indices,
            self.gripper_flat_corner_ids,
            self.gripper_flat_offsets,
            self.gripper_patch_sizes,
        ) = self._compute_gripper_patches(
            builder_particle_positions,
            cloth_particle_start=self.cloth_particle_start,
            gripper_size=self.args.gripper_size,
        )

        # Prescribed boundary DOFs must not move inside Newton's nonlinear
        # solve; repinning otherwise only conceals their intermediate motion.
        for idx in self.gripper_flat_indices:
            builder.particle_flags[int(idx)] &= ~int(newton.ParticleFlags.ACTIVE)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_radius = self.args.contact_radius
        self.model.soft_contact_margin = self.args.contact_margin
        self.model.soft_contact_ke = self.args.contact_ke
        self.model.soft_contact_kd = self.args.contact_kd
        self.model.soft_contact_mu = self.args.contact_mu

        self.solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
            linear_iterations=self.linear_iterations,
        )
        # Current SolverStyle3D precomputes from the finalized model in __init__.

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.device = self.state_0.particle_q.device
        self.initial_corner_positions = self.state_0.particle_q.numpy()[self.corner_indices].astype(np.float32, copy=True)
        self._cached_cloth_particles = self._read_cloth_particles_from_state()
        self.corner_target_positions = np.array(self.initial_corner_positions, copy=True)
        self._apply_initial_follower_target_offset()
        self.corner_target_velocities = np.zeros((4, 3), dtype=np.float32)

        self.corner_targets_wp = wp.array(self.corner_target_positions, dtype=wp.vec3, device=self.device)
        self.gripper_flat_indices_wp = wp.array(self.gripper_flat_indices, dtype=int, device=self.device)
        self.gripper_flat_corner_ids_wp = wp.array(self.gripper_flat_corner_ids, dtype=int, device=self.device)
        self.gripper_flat_offsets_wp = wp.array(self.gripper_flat_offsets, dtype=wp.vec3, device=self.device)

        self.motion = ClothRuntime(self.model, self.gripper_flat_indices,
                                   self.gripper_flat_corner_ids, self.gripper_flat_offsets,
                                   self.sim_substeps, self.sim_dt, self.iterations,
                                   self.linear_iterations, solver=self.solver,
                                   capture=self.use_cuda_graph)
        self.state_0, self.state_1 = self.motion.state, self.motion.scratch

    def capture(self):
        # ClothRuntime captures with saved/restored state, including odd steps.
        self.graph = self.motion.graph

    def simulate(self):
        self.motion._advance()

    def _integrate_corner_targets(self):
        self.motion.starts.assign(self.corner_target_positions)
        self.motion.commands.assign(self.corner_target_velocities)
        self.corner_target_positions += self.corner_target_velocities*self.frame_dt
        self.corner_targets_wp.assign(self.corner_target_positions)

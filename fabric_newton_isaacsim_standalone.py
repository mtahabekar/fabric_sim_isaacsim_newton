#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys


_CONTROLLER_ARG_VALUE_COUNTS = {
    "--grid-x": 1,
    "--grid-y": 1,
    "--cell-x": 1,
    "--cell-y": 1,
    "--cloth-start-z": 1,
    "--mass-per-particle": 1,
    "--particle-radius": 1,
    "--table-hx": 1,
    "--table-hy": 1,
    "--table-hz": 1,
    "--gripper-size": 1,
    "--substeps": 1,
    "--iterations": 1,
    "--linear-iterations": 1,
    "--state-publish-every": 1,
    "--tri-ke-weft": 1,
    "--tri-ke-warp": 1,
    "--tri-ke-shear": 1,
    "--tri-ka": 1,
    "--tri-kd": 1,
    "--edge-ke-weft": 1,
    "--edge-ke-warp": 1,
    "--edge-ke-shear": 1,
    "--edge-kd": 1,
    "--contact-radius": 1,
    "--contact-margin": 1,
    "--contact-ke": 1,
    "--contact-kd": 1,
    "--contact-mu": 1,
    "--ros-frame-id": 1,
    "--cmd-timeout": 1,
    "--bottom-command-max-speed": 1,
    "--disable-cuda-graph": 0,
    "--external-cbf-enabled": 0,
    "--cbf-command-timeout": 1,
    "--cbf-stale-policy": 1,
}


def _argv_without_controller_args(argv: list[str]) -> list[str]:
    filtered = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        name = arg.split("=", 1)[0]
        value_count = _CONTROLLER_ARG_VALUE_COUNTS.get(name)
        if value_count is None:
            filtered.append(arg)
            i += 1
            continue

        if "=" in arg:
            i += 1
        else:
            i += 1 + value_count
    return filtered


_ORIGINAL_ARGV = sys.argv[:]
sys.argv = _argv_without_controller_args(sys.argv)

from isaacsim import SimulationApp

# Must be created before Omniverse imports.
simulation_app = SimulationApp({"headless": False})

import argparse
import time
import traceback

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux
import warp as wp

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
sys.argv = _ORIGINAL_ARGV

import rclpy
from geometry_msgs.msg import Pose, PoseArray, TwistStamped
from sensor_msgs.msg import PointCloud2, PointField

import newton
from newton.solvers import style3d


IDX_BOTTOM_LEFT = 0
IDX_BOTTOM_RIGHT = 1
IDX_TOP_LEFT = 2
IDX_TOP_RIGHT = 3


@wp.kernel
def pin_gripper_particles(
    patch_indices: wp.array(dtype=int),
    patch_corner_ids: wp.array(dtype=int),
    patch_offsets: wp.array(dtype=wp.vec3),
    corner_targets: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Pin every particle in every gripper patch rigidly to its corner target."""
    tid = wp.tid()
    corner_id = patch_corner_ids[tid]
    particle_idx = patch_indices[tid]
    particle_q[particle_idx] = corner_targets[corner_id] + patch_offsets[tid]
    particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)


def set_stage():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    light.CreateIntensityAttr(3000.0)

    return stage


def create_visual_ground(stage):
    mesh = UsdGeom.Mesh.Define(stage, "/World/Ground")
    size = 2.5
    mesh.CreatePointsAttr([
        Gf.Vec3f(-size, -size, 0.0),
        Gf.Vec3f( size, -size, 0.0),
        Gf.Vec3f( size,  size, 0.0),
        Gf.Vec3f(-size,  size, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorPrimvar("constant").Set([Gf.Vec3f(0.45, 0.45, 0.45)])


def create_visual_box(stage, prim_path: str, center_xyz, half_extents_xyz, color=(0.2, 0.25, 0.35)):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorPrimvar("constant").Set([Gf.Vec3f(*color)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])))
    xform.AddScaleOp().Set(
        Gf.Vec3f(
            float(2.0 * half_extents_xyz[0]),
            float(2.0 * half_extents_xyz[1]),
            float(2.0 * half_extents_xyz[2]),
        )
    )


def create_cloth_mesh(stage, prim_path: str, particle_positions: np.ndarray):
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorPrimvar("constant").Set([Gf.Vec3f(0.75, 0.15, 0.15)])

    xs = np.unique(np.round(particle_positions[:, 0], decimals=6))
    ys = np.unique(np.round(particle_positions[:, 1], decimals=6))

    x_to_i = {float(x): i for i, x in enumerate(xs)}
    y_to_i = {float(y): i for i, y in enumerate(ys)}

    grid = -np.ones((len(ys), len(xs)), dtype=np.int32)
    rounded_xy = np.round(particle_positions[:, :2], decimals=6)

    for idx, (x, y) in enumerate(rounded_xy):
        grid[y_to_i[float(y)], x_to_i[float(x)]] = idx

    if np.any(grid < 0):
        raise RuntimeError("Failed to reconstruct cloth grid topology from initial particle positions.")

    face_counts = []
    face_indices = []
    ny, nx = grid.shape

    for j in range(ny - 1):
        for i in range(nx - 1):
            a = int(grid[j, i])
            b = int(grid[j, i + 1])
            c = int(grid[j + 1, i])
            d = int(grid[j + 1, i + 1])

            face_counts.extend([3, 3])
            face_indices.extend([a, b, d])
            face_indices.extend([a, d, c])

    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in particle_positions])

    return mesh


class FabricRosIo:
    def __init__(
        self,
        frame_id: str = "world",
        cmd_timeout: float = 0.25,
        cbf_cmd_timeout: float = 0.10,
    ):
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node("style3d_fabric_velocity_controller")
        self.frame_id = frame_id
        self.cmd_timeout = float(cmd_timeout)
        self.cbf_cmd_timeout = float(cbf_cmd_timeout)

        self.latest_top_left_cmd_vel = np.zeros(3, dtype=np.float32)
        self.latest_top_right_cmd_vel = np.zeros(3, dtype=np.float32)
        self.latest_top_left_cmd_time: float | None = None
        self.latest_top_right_cmd_time: float | None = None
        self.latest_bottom_left_cbf_vel = np.zeros(3, dtype=np.float32)
        self.latest_bottom_right_cbf_vel = np.zeros(3, dtype=np.float32)
        self.latest_bottom_left_cbf_time: float | None = None
        self.latest_bottom_right_cbf_time: float | None = None

        self.sub_top_left_cmd = self.node.create_subscription(
            TwistStamped,
            "/fabric/top_left/cmd_vel",
            self._top_left_cmd_cb,
            10,
        )
        self.sub_top_right_cmd = self.node.create_subscription(
            TwistStamped,
            "/fabric/top_right/cmd_vel",
            self._top_right_cmd_cb,
            10,
        )
        self.sub_bottom_left_cbf_cmd = self.node.create_subscription(
            TwistStamped,
            "/fabric/bottom_left/cbf_cmd_vel",
            self._bottom_left_cbf_cb,
            10,
        )
        self.sub_bottom_right_cbf_cmd = self.node.create_subscription(
            TwistStamped,
            "/fabric/bottom_right/cbf_cmd_vel",
            self._bottom_right_cbf_cb,
            10,
        )

        self.pub_left_robot_cmd = self.node.create_publisher(
            TwistStamped,
            "/robot_left/ee_cmd_vel",
            10,
        )
        self.pub_right_robot_cmd = self.node.create_publisher(
            TwistStamped,
            "/robot_right/ee_cmd_vel",
            10,
        )
        self.pub_top_left_cbf_input_cmd = self.node.create_publisher(
            TwistStamped,
            "/fabric/top_left/cbf_input_cmd_vel",
            10,
        )
        self.pub_top_right_cbf_input_cmd = self.node.create_publisher(
            TwistStamped,
            "/fabric/top_right/cbf_input_cmd_vel",
            10,
        )
        self.pub_bottom_left_applied_cmd = self.node.create_publisher(
            TwistStamped,
            "/fabric/bottom_left/applied_cmd_vel",
            10,
        )
        self.pub_bottom_right_applied_cmd = self.node.create_publisher(
            TwistStamped,
            "/fabric/bottom_right/applied_cmd_vel",
            10,
        )
        self.pub_sim_fabric_corners = self.node.create_publisher(
            PoseArray,
            "/fabric/sim/corners",
            10,
        )
        self.pub_sim_fabric_state = self.node.create_publisher(
            PointCloud2,
            "/fabric/sim/state",
            10,
        )

        self.node.get_logger().info(
            "ROS2 bridge I/O ready: subscribing to /fabric/top_left/cmd_vel and "
            "/fabric/top_right/cmd_vel plus /fabric/bottom_left/cbf_cmd_vel and "
            "/fabric/bottom_right/cbf_cmd_vel; publishing top CBF input commands, "
            "applied bottom commands, /robot_left/ee_cmd_vel, "
            "/robot_right/ee_cmd_vel, /fabric/sim/corners, and /fabric/sim/state."
        )

    def _top_left_cmd_cb(self, msg: TwistStamped) -> None:
        self.latest_top_left_cmd_vel[:] = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
        )
        self.latest_top_left_cmd_time = time.monotonic()

    def _top_right_cmd_cb(self, msg: TwistStamped) -> None:
        self.latest_top_right_cmd_vel[:] = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
        )
        self.latest_top_right_cmd_time = time.monotonic()

    def _bottom_left_cbf_cb(self, msg: TwistStamped) -> None:
        self.latest_bottom_left_cbf_vel[:] = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
        )
        self.latest_bottom_left_cbf_time = time.monotonic()

    def _bottom_right_cbf_cb(self, msg: TwistStamped) -> None:
        self.latest_bottom_right_cbf_vel[:] = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
        )
        self.latest_bottom_right_cbf_time = time.monotonic()

    def spin_once(self) -> None:
        # A tiny bounded wait avoids a zero-time rclpy poll during Kit/ROS startup,
        # while still keeping ROS callbacks inside the simulation frame loop.
        rclpy.spin_once(self.node, timeout_sec=0.001)

    def _get_recent_cmd_with_timeout(
        self,
        vel: np.ndarray,
        stamp: float | None,
        timeout: float,
    ) -> np.ndarray:
        if stamp is None or time.monotonic() - stamp > timeout:
            return np.zeros(3, dtype=np.float32)
        return np.array(vel, dtype=np.float32, copy=True)

    def _get_recent_cmd(self, vel: np.ndarray, stamp: float | None) -> np.ndarray:
        return self._get_recent_cmd_with_timeout(vel, stamp, self.cmd_timeout)

    def get_top_left_cmd_vel(self) -> np.ndarray:
        return self._get_recent_cmd(self.latest_top_left_cmd_vel, self.latest_top_left_cmd_time)

    def get_top_right_cmd_vel(self) -> np.ndarray:
        return self._get_recent_cmd(self.latest_top_right_cmd_vel, self.latest_top_right_cmd_time)

    def get_bottom_cbf_cmd_vels(self) -> tuple[bool, np.ndarray]:
        now = time.monotonic()
        left_fresh = (
            self.latest_bottom_left_cbf_time is not None
            and now - self.latest_bottom_left_cbf_time <= self.cbf_cmd_timeout
        )
        right_fresh = (
            self.latest_bottom_right_cbf_time is not None
            and now - self.latest_bottom_right_cbf_time <= self.cbf_cmd_timeout
        )
        if not (left_fresh and right_fresh):
            return False, np.zeros((2, 3), dtype=np.float32)

        safe_vels = np.array(
            [
                self.latest_bottom_left_cbf_vel,
                self.latest_bottom_right_cbf_vel,
            ],
            dtype=np.float32,
            copy=True,
        )
        return True, safe_vels

    def _make_twist(self, vel_xyz: np.ndarray) -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.twist.linear.x = float(vel_xyz[0])
        msg.twist.linear.y = float(vel_xyz[1])
        msg.twist.linear.z = float(vel_xyz[2])
        return msg

    def publish_robot_velocity_commands(
        self,
        left_vel_xyz: np.ndarray,
        right_vel_xyz: np.ndarray,
    ) -> None:
        self.pub_left_robot_cmd.publish(self._make_twist(left_vel_xyz))
        self.pub_right_robot_cmd.publish(self._make_twist(right_vel_xyz))

    def publish_top_cbf_input_velocity_commands(
        self,
        top_left_vel_xyz: np.ndarray,
        top_right_vel_xyz: np.ndarray,
    ) -> None:
        self.pub_top_left_cbf_input_cmd.publish(self._make_twist(top_left_vel_xyz))
        self.pub_top_right_cbf_input_cmd.publish(self._make_twist(top_right_vel_xyz))

    def publish_bottom_applied_velocity_commands(
        self,
        left_vel_xyz: np.ndarray,
        right_vel_xyz: np.ndarray,
    ) -> None:
        self.pub_bottom_left_applied_cmd.publish(self._make_twist(left_vel_xyz))
        self.pub_bottom_right_applied_cmd.publish(self._make_twist(right_vel_xyz))

    def _make_pose_array(self, positions: np.ndarray) -> PoseArray:
        msg = PoseArray()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()

        for p in positions:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg

    def publish_simulated_fabric_corners(self, corner_positions: np.ndarray) -> None:
        self.pub_sim_fabric_corners.publish(self._make_pose_array(corner_positions))

    def publish_simulated_fabric_state(self, particle_positions: np.ndarray) -> None:
        points = np.asarray(particle_positions, dtype=np.float32)
        points = np.ascontiguousarray(points[:, :3])

        msg = PointCloud2()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        self.pub_sim_fabric_state.publish(msg)

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


class Style3DIsaacStandalone:
    def __init__(self, args):
        self.args = args
        self.sim_time = 0.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = args.iterations
        self.linear_iterations = args.linear_iterations
        self.state_publish_every = max(1, int(args.state_publish_every))
        self._frame_index = 0

        self.grid_cells_x = args.grid_x
        self.grid_cells_y = args.grid_y
        self.cell_x = args.cell_x
        self.cell_y = args.cell_y
        self.cloth_width = self.grid_cells_x * self.cell_x
        self.cloth_height = self.grid_cells_y * self.cell_y
        self.cloth_particle_radius = args.particle_radius
        self.cloth_start_z = args.cloth_start_z

        self.table_half_extents = np.array([args.table_hx, args.table_hy, args.table_hz], dtype=np.float32)
        self.table_center = wp.vec3(0.0, 0.0, float(args.table_hz))

        self.corner_names = ("bottom_left", "bottom_right", "top_left", "top_right")
        self.fixed_corner_names = ("top_left", "top_right")
        self.moving_corner_names = ("bottom_left", "bottom_right")
        self.bottom_command_max_speed = float(args.bottom_command_max_speed)
        self.use_cuda_graph = not bool(args.disable_cuda_graph)
        self.external_cbf_enabled = bool(args.external_cbf_enabled)
        self.cbf_stale_policy = str(args.cbf_stale_policy)
        self.last_applied_bottom_vels = np.zeros((2, 3), dtype=np.float32)

        self.stage = set_stage()
        create_visual_ground(self.stage)
        create_visual_box(
            self.stage,
            "/World/Table",
            center_xyz=(0.0, 0.0, float(args.table_hz)),
            half_extents_xyz=(float(args.table_hx), float(args.table_hy), float(args.table_hz)),
        )

        self._build_model()
        self._create_usd_debug_prims()
        self._print_startup_summary()
        self.capture()
        self.ros = None

    def initialize_ros_io(self) -> None:
        if self.ros is not None:
            return

        self.ros = FabricRosIo(
            frame_id=self.args.ros_frame_id,
            cmd_timeout=self.args.cmd_timeout,
            cbf_cmd_timeout=self.args.cbf_command_timeout,
        )

    def _build_model(self):
        cloth_origin = wp.vec3(-0.5 * self.cloth_width, -0.5 * self.cloth_height, self.cloth_start_z)

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
        self.solver._precompute(builder)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.device = self.state_0.particle_q.device
        self.initial_corner_positions = self.state_0.particle_q.numpy()[self.corner_indices].astype(np.float32, copy=True)
        self._cached_cloth_particles = self._read_cloth_particles_from_state()
        self.corner_target_positions = np.array(self.initial_corner_positions, copy=True)
        self.corner_target_velocities = np.zeros((4, 3), dtype=np.float32)

        self.corner_targets_wp = wp.array(self.corner_target_positions, dtype=wp.vec3, device=self.device)
        self.gripper_flat_indices_wp = wp.array(self.gripper_flat_indices, dtype=int, device=self.device)
        self.gripper_flat_corner_ids_wp = wp.array(self.gripper_flat_corner_ids, dtype=int, device=self.device)
        self.gripper_flat_offsets_wp = wp.array(self.gripper_flat_offsets, dtype=wp.vec3, device=self.device)

    def _compute_gripper_patches(
        self,
        particle_positions: np.ndarray,
        cloth_particle_start: int,
        gripper_size: int,
    ) -> tuple:
        xs = np.unique(np.round(particle_positions[:, 0], decimals=6))
        ys = np.unique(np.round(particle_positions[:, 1], decimals=6))
        expected_x_count = self.grid_cells_x + 1
        expected_y_count = self.grid_cells_y + 1

        if len(xs) != expected_x_count or len(ys) != expected_y_count:
            raise RuntimeError(
                f"Unexpected grid layout from Style3D cloth builder: got {len(xs)} x-values "
                f"and {len(ys)} y-values, expected {expected_x_count} and {expected_y_count}."
            )

        min_x, max_x = float(xs[0]), float(xs[-1])
        min_y, max_y = float(ys[0]), float(ys[-1])

        eps_x = 0.5 * self.cell_x
        eps_y = 0.5 * self.cell_y

        reach_x = (gripper_size - 1) * self.cell_x + eps_x
        reach_y = (gripper_size - 1) * self.cell_y + eps_y

        corner_defs = {
            "bottom_left":  (
                np.array([min_x, min_y]),
                lambda px: px[:, 0] <= min_x + reach_x,
                lambda py: py[:, 1] <= min_y + reach_y,
            ),
            "bottom_right": (
                np.array([max_x, min_y]),
                lambda px: px[:, 0] >= max_x - reach_x,
                lambda py: py[:, 1] <= min_y + reach_y,
            ),
            "top_left": (
                np.array([min_x, max_y]),
                lambda px: px[:, 0] <= min_x + reach_x,
                lambda py: py[:, 1] >= max_y - reach_y,
            ),
            "top_right": (
                np.array([max_x, max_y]),
                lambda px: px[:, 0] >= max_x - reach_x,
                lambda py: py[:, 1] >= max_y - reach_y,
            ),
        }

        corner_indices = []
        flat_global = []
        flat_cids = []
        flat_offs = []
        patch_sizes = []
        used_tip_locals = set()

        for cid, corner_name in enumerate(self.corner_names):
            tip_xy, sel_x, sel_y = corner_defs[corner_name]

            xy = particle_positions[:, :2]
            dists = np.linalg.norm(xy - tip_xy, axis=1)
            tip_local = int(np.argmin(dists))
            tip_tol = 0.25 * min(self.cell_x, self.cell_y)

            if tip_local in used_tip_locals:
                raise RuntimeError(f"Duplicate tip particle for {corner_name}.")
            if float(dists[tip_local]) > tip_tol:
                raise RuntimeError(
                    f"Cannot locate tip particle for {corner_name} "
                    f"(dist={dists[tip_local]:.6f} > tol={tip_tol:.6f})."
                )

            used_tip_locals.add(tip_local)
            tip_global = cloth_particle_start + tip_local
            corner_indices.append(tip_global)
            tip_pos = particle_positions[tip_local]

            mask = sel_x(particle_positions) & sel_y(particle_positions)
            local_patch_indices = np.where(mask)[0]
            if len(local_patch_indices) == 0:
                raise RuntimeError(f"No particles found in gripper patch for {corner_name}.")

            for li in local_patch_indices:
                flat_global.append(cloth_particle_start + int(li))
                flat_cids.append(cid)
                offset = particle_positions[li] - tip_pos
                flat_offs.append(offset.astype(np.float32))

            patch_sizes.append(len(local_patch_indices))

        return (
            corner_indices,
            np.array(flat_global, dtype=np.int32),
            np.array(flat_cids, dtype=np.int32),
            np.array(flat_offs, dtype=np.float32),
            patch_sizes,
        )

    def _print_startup_summary(self) -> None:
        print("Isaac Sim standalone + Newton Style3D cloth simulation")
        print(f"Python executable: {sys.executable}")
        print(
            "Cloth grid: "
            f"{self.grid_cells_x}x{self.grid_cells_y} cells, "
            f"{self.grid_cells_x + 1}x{self.grid_cells_y + 1} particles, "
            f"cell=({self.cell_x:.3f}, {self.cell_y:.3f}) m"
        )
        print(
            "Table box: "
            f"center=(0.000, 0.000, {float(self.table_half_extents[2]):.3f}) m, "
            f"half_extents=({float(self.table_half_extents[0]):.3f}, "
            f"{float(self.table_half_extents[1]):.3f}, {float(self.table_half_extents[2]):.3f}) m"
        )
        print(f"Solver dt: frame_dt={self.frame_dt:.6f}, substeps={self.sim_substeps}, sim_dt={self.sim_dt:.6f}")
        print(f"Solver iterations: iterations={self.iterations}, linear_iterations={self.linear_iterations}")
        print(f"State publish: every {self.state_publish_every} frame(s)")
        print(
            "ROS2 velocity control: "
            "/fabric/top_left/cmd_vel, /fabric/top_right/cmd_vel -> integrated top targets; "
            "top CBF input topics, bottom applied command topics, and robot ee_cmd_vel publishers"
        )
        print(
            "Bottom command limiting: "
            f"max_speed={self.bottom_command_max_speed:.3f} m/s "
            f"({'disabled' if self.bottom_command_max_speed <= 0.0 else 'enabled'}), "
            "publishing fabric state on /fabric/sim/state"
        )
        print(
            "External CBF interface: "
            f"{'enabled' if self.external_cbf_enabled else 'disabled'}, "
            f"command_timeout={self.args.cbf_command_timeout:.3f} s, "
            f"stale_policy={self.cbf_stale_policy}"
        )
        print(
            "External CBF input topics: "
            "/fabric/top_left/cbf_input_cmd_vel, /fabric/top_right/cbf_input_cmd_vel"
        )
        print(
            "External CBF output topics: "
            "/fabric/bottom_left/cbf_cmd_vel, /fabric/bottom_right/cbf_cmd_vel"
        )
        print(
            "Bottom applied command topics: "
            "/fabric/bottom_left/applied_cmd_vel, /fabric/bottom_right/applied_cmd_vel"
        )

        total_pinned = sum(self.gripper_patch_sizes)
        print(
            f"Gripper patches: {self.args.gripper_size}x{self.args.gripper_size} per corner "
            f"({total_pinned} particles pinned total)"
        )

        for corner_name, corner_index, corner_pos, patch_n in zip(
            self.corner_names,
            self.corner_indices,
            self.initial_corner_positions,
            self.gripper_patch_sizes,
            strict=True,
        ):
            print(
                f"  {corner_name}: tip particle {corner_index}, "
                f"patch={patch_n} particles, "
                f"initial_pos=({corner_pos[0]:.4f}, {corner_pos[1]:.4f}, {corner_pos[2]:.4f})"
            )
        print()

    def _create_usd_debug_prims(self):
        current_particles = self.get_cloth_particles()
        self.cloth_mesh = create_cloth_mesh(self.stage, "/World/Cloth", current_particles)

        self.target_corner_markers = []
        self.actual_corner_markers = []

        target_colors = [
            (1.0, 0.2, 0.2),
            (0.2, 1.0, 0.2),
            (0.2, 0.2, 1.0),
            (1.0, 1.0, 0.2),
        ]
        actual_colors = [
            (0.6, 0.0, 0.0),
            (0.0, 0.6, 0.0),
            (0.0, 0.0, 0.6),
            (0.6, 0.6, 0.0),
        ]

        for i in range(4):
            target_path = f"/World/CornerTarget_{i}"
            target_sphere = UsdGeom.Sphere.Define(self.stage, target_path)
            target_sphere.CreateRadiusAttr(0.02)
            target_sphere.CreateDisplayColorPrimvar("constant").Set([Gf.Vec3f(*target_colors[i])])
            target_xf = UsdGeom.Xformable(target_sphere.GetPrim())
            p = self.initial_corner_positions[i]
            target_xf.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            self.target_corner_markers.append(target_xf)

            actual_path = f"/World/CornerActual_{i}"
            actual_sphere = UsdGeom.Sphere.Define(self.stage, actual_path)
            actual_sphere.CreateRadiusAttr(0.014)
            actual_sphere.CreateDisplayColorPrimvar("constant").Set([Gf.Vec3f(*actual_colors[i])])
            actual_xf = UsdGeom.Xformable(actual_sphere.GetPrim())
            actual_xf.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            self.actual_corner_markers.append(actual_xf)

    def _get_actual_corner_positions(self) -> np.ndarray:
        full_particle_q = self.state_0.particle_q.numpy()
        return np.asarray(full_particle_q[self.corner_indices], dtype=np.float32)

    def _apply_hard_pins(self, state: newton.State, corner_targets_wp=None) -> None:
        if corner_targets_wp is None:
            corner_targets_wp = self.corner_targets_wp

        wp.launch(
            pin_gripper_particles,
            dim=len(self.gripper_flat_indices),
            inputs=[
                self.gripper_flat_indices_wp,
                self.gripper_flat_corner_ids_wp,
                self.gripper_flat_offsets_wp,
                corner_targets_wp,
                state.particle_q,
                state.particle_qd,
            ],
            device=self.device,
        )

    def _apply_external_forces(self, state: newton.State) -> None:
        # Placeholder for future ROS or user-injected forces.
        # The original Newton viewer path called viewer.apply_forces(state).
        # In Isaac Sim standalone there is no Newton viewer object, so this is a no-op.
        return

    def capture(self) -> None:
        if not self.use_cuda_graph:
            print("CUDA graph disabled by startup argument.")
            self.graph = None
            return

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_external_forces(self.state_0)

            self._apply_hard_pins(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self._apply_hard_pins(self.state_1)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def _get_top_corner_velocity_commands(self) -> np.ndarray:
        """Read ROS2 top-corner velocity commands.

        Returns shape (2, 3): [top_left_velocity, top_right_velocity].
        Commands are velocities in m/s; they are never interpreted as positions.
        """
        if self.ros is None:
            return np.zeros((2, 3), dtype=np.float32)

        top_cmd_vels = np.zeros((2, 3), dtype=np.float32)
        top_cmd_vels[0] = self.ros.get_top_left_cmd_vel()
        top_cmd_vels[1] = self.ros.get_top_right_cmd_vel()
        return top_cmd_vels

    def _limit_velocity_norms(self, velocities: np.ndarray, max_speed: float) -> np.ndarray:
        if max_speed <= 0.0:
            return velocities

        limited = np.array(velocities, dtype=np.float32, copy=True)
        norms = np.linalg.norm(limited, axis=1)
        for i, norm in enumerate(norms):
            if norm > max_speed:
                limited[i] *= max_speed / norm
        return limited

    def _compute_bottom_applied_velocities(
        self,
        top_cmd_vels: np.ndarray,
    ) -> np.ndarray:
        nominal = np.asarray(top_cmd_vels, dtype=np.float32).reshape(2, 3).copy()
        if not self.external_cbf_enabled:
            applied = nominal
        elif self.ros is not None:
            ok, safe_vels = self.ros.get_bottom_cbf_cmd_vels()
            if ok:
                applied = np.asarray(safe_vels, dtype=np.float32).reshape(2, 3)
            else:
                applied = self._stale_bottom_velocity_fallback(nominal)
        else:
            applied = self._stale_bottom_velocity_fallback(nominal)

        applied = np.asarray(applied, dtype=np.float32).reshape(2, 3).copy()
        applied = self._limit_velocity_norms(applied, self.bottom_command_max_speed)
        self.last_applied_bottom_vels = applied.copy()
        return applied

    def _stale_bottom_velocity_fallback(self, nominal: np.ndarray) -> np.ndarray:
        if self.cbf_stale_policy == "zero":
            return np.zeros_like(nominal)
        if self.cbf_stale_policy == "hold-last":
            return self.last_applied_bottom_vels
        return nominal

    def _assemble_corner_target_velocities(
        self,
        top_cmd_vels: np.ndarray,
        bottom_applied_vels: np.ndarray,
    ) -> np.ndarray:
        vels = np.zeros((4, 3), dtype=np.float32)
        vels[IDX_BOTTOM_LEFT] = bottom_applied_vels[0]
        vels[IDX_BOTTOM_RIGHT] = bottom_applied_vels[1]
        vels[IDX_TOP_LEFT] = top_cmd_vels[0]
        vels[IDX_TOP_RIGHT] = top_cmd_vels[1]
        return vels

    def _integrate_corner_targets(self) -> None:
        self.corner_target_positions += self.corner_target_velocities * self.frame_dt
        self.corner_targets_wp.assign(self.corner_target_positions)

    def step(self) -> None:
        if self.ros is None:
            raise RuntimeError("ROS I/O must be initialized before stepping the controller.")

        self.ros.spin_once()

        top_cmd_vels = self._get_top_corner_velocity_commands()
        self.ros.publish_top_cbf_input_velocity_commands(
            top_left_vel_xyz=top_cmd_vels[0],
            top_right_vel_xyz=top_cmd_vels[1],
        )

        bottom_applied_vels = self._compute_bottom_applied_velocities(top_cmd_vels)
        self.ros.publish_bottom_applied_velocity_commands(
            left_vel_xyz=bottom_applied_vels[0],
            right_vel_xyz=bottom_applied_vels[1],
        )

        self.ros.publish_robot_velocity_commands(
            left_vel_xyz=bottom_applied_vels[0],
            right_vel_xyz=bottom_applied_vels[1],
        )

        self.corner_target_velocities = self._assemble_corner_target_velocities(
            top_cmd_vels,
            bottom_applied_vels,
        )
        self._integrate_corner_targets()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self._update_usd_visuals()
        self.ros.publish_simulated_fabric_corners(self._get_actual_corner_positions())
        if self._frame_index % self.state_publish_every == 0:
            self.ros.publish_simulated_fabric_state(self._cached_cloth_particles)
        self._frame_index += 1

    def test_final(self) -> None:
        particle_positions = self.state_0.particle_q.numpy()
        if float(np.min(particle_positions[:, 2])) < -0.05:
            raise ValueError("Cloth particles went too far below the ground plane during the smoke test.")

        final_corner_positions = particle_positions[self.corner_indices]
        max_corner_error = float(np.max(np.linalg.norm(final_corner_positions - self.corner_target_positions, axis=1)))
        if max_corner_error > 1.0e-4:
            raise ValueError(f"Hard-pinned corners drifted from targets by {max_corner_error:.6e} m.")

    def get_cloth_particles(self) -> np.ndarray:
        particles = self._read_cloth_particles_from_state()
        self._cached_cloth_particles = particles
        return particles

    def _read_cloth_particles_from_state(self) -> np.ndarray:
        q = self.state_0.particle_q.numpy()
        return np.asarray(q[self.cloth_particle_start:self.cloth_particle_end], dtype=np.float32)

    def _update_usd_visuals(self):
        cloth_particles = self.get_cloth_particles()
        self.cloth_mesh.GetPointsAttr().Set(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in cloth_particles]
        )

        full_particle_q = self.state_0.particle_q.numpy()
        actual_corner_positions = np.asarray(full_particle_q[self.corner_indices], dtype=np.float32)

        for i in range(4):
            tgt = self.corner_target_positions[i]
            act = actual_corner_positions[i]

            tgt_ops = self.target_corner_markers[i].GetOrderedXformOps()
            tgt_ops[0].Set(Gf.Vec3d(float(tgt[0]), float(tgt[1]), float(tgt[2])))

            act_ops = self.actual_corner_markers[i].GetOrderedXformOps()
            act_ops[0].Set(Gf.Vec3d(float(act[0]), float(act[1]), float(act[2])))

    def shutdown(self) -> None:
        if self.ros is not None:
            self.ros.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grid-x", type=int, default=56, help="Cloth cells along X.")
    parser.add_argument("--grid-y", type=int, default=56, help="Cloth cells along Y.")
    parser.add_argument("--cell-x", type=float, default=0.015, help="Cloth cell size along X in meters.")
    parser.add_argument("--cell-y", type=float, default=0.015, help="Cloth cell size along Y in meters.")
    parser.add_argument("--cloth-start-z", type=float, default=0.72, help="Initial cloth height in meters.")
    parser.add_argument("--mass-per-particle", type=float, default=0.003, help="Style3D mass per particle.")
    parser.add_argument("--particle-radius", type=float, default=0.008, help="Cloth particle radius in meters.")

    parser.add_argument("--table-hx", type=float, default=0.22, help="Table half-size along X.")
    parser.add_argument("--table-hy", type=float, default=0.22, help="Table half-size along Y.")
    parser.add_argument("--table-hz", type=float, default=0.16, help="Table half-size along Z.")

    parser.add_argument(
        "--gripper-size",
        type=int,
        default=3,
        help="Square gripper patch side length in particles (e.g. 3 -> 3x3=9 particles per corner).",
    )
    parser.add_argument("--substeps", type=int, default=8, help="Simulation substeps per rendered frame.")
    parser.add_argument("--iterations", type=int, default=10, help="Style3D outer iterations.")
    parser.add_argument("--linear-iterations", type=int, default=16, help="Style3D linear solver iterations.")
    parser.add_argument(
        "--state-publish-every",
        type=int,
        default=3,
        help="Publish /fabric/sim/state every N simulation frames.",
    )

    parser.add_argument("--tri-ke-weft", type=float, default=3.0e4, help="Weft stretch stiffness.")
    parser.add_argument("--tri-ke-warp", type=float, default=3.0e4, help="Warp stretch stiffness.")
    parser.add_argument("--tri-ke-shear", type=float, default=6.0e3, help="Shear stretch stiffness.")
    parser.add_argument("--tri-ka", type=float, default=5.0e3, help="Triangle area stiffness.")
    parser.add_argument("--tri-kd", type=float, default=5.0e-5, help="Triangle damping.")

    parser.add_argument("--edge-ke-weft", type=float, default=2.0e-3, help="Weft bending stiffness.")
    parser.add_argument("--edge-ke-warp", type=float, default=1.0e-3, help="Warp bending stiffness.")
    parser.add_argument("--edge-ke-shear", type=float, default=5.0e-4, help="Shear bending stiffness.")
    parser.add_argument("--edge-kd", type=float, default=1.0e-5, help="Edge damping.")

    parser.add_argument(
        "--contact-radius",
        type=float,
        default=0.01,
        help="Soft contact query radius in meters (should be >= particle-radius).",
    )
    parser.add_argument("--contact-margin", type=float, default=0.012, help="Soft contact margin in meters.")
    parser.add_argument("--contact-ke", type=float, default=4.0e3, help="Soft contact stiffness.")
    parser.add_argument("--contact-kd", type=float, default=1.0e-2, help="Soft contact damping.")
    parser.add_argument("--contact-mu", type=float, default=2.0, help="Soft contact friction.")

    parser.add_argument("--ros-frame-id", type=str, default="world", help="Frame id used on ROS2 messages.")
    parser.add_argument(
        "--cmd-timeout",
        type=float,
        default=1.0,
        help="Seconds before stale top-corner velocity commands are treated as zero.",
    )
    parser.add_argument(
        "--bottom-command-max-speed",
        type=float,
        default=0.0,
        help="Maximum applied bottom-corner command speed in m/s; <= 0 disables limiting.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable Warp CUDA graph capture for the live simulation.",
    )
    parser.add_argument(
        "--external-cbf-enabled",
        action="store_true",
        default=False,
        help="Use externally supplied safe bottom-corner CBF velocity commands when fresh.",
    )
    parser.add_argument(
        "--cbf-command-timeout",
        type=float,
        default=0.10,
        help="Seconds before external CBF safe commands are considered stale.",
    )
    parser.add_argument(
        "--cbf-stale-policy",
        choices=("nominal", "zero", "hold-last"),
        default="nominal",
        help="Fallback when external CBF commands are missing or stale.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    sim = None

    try:
        sim = Style3DIsaacStandalone(args)

        # Let Kit render/ROS extensions settle before the first cloth step mutates USD.
        simulation_app.update()
        sim.initialize_ros_io()

        for _ in range(3):
            sim.step()
            simulation_app.update()

        while simulation_app.is_running():
            sim.step()
            simulation_app.update()

    finally:
        if sim is not None:
            try:
                sim.shutdown()
            except Exception:
                traceback.print_exc()
        simulation_app.close()


if __name__ == "__main__":
    main()

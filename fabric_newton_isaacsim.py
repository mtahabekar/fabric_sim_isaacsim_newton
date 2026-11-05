#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim + Newton Style3D fabric simulation with the sensitivity CBF.

Four square gripper patches hold the cloth corners. The two leaders
(corners 2, 3) follow ROS 2 velocity commands. Every frame the two followers
(corners 0, 1) receive the leader-tracking formation command filtered by the
sensitivity CBF (capsule_cbf/sensitivity_cbf.py): the cloth is kept outside a
capsule that encloses the table, and gripper tension and fabric strain are
limited. The filter runs synchronously inside the frame on the same particle
state that is about to be advanced.

Run through `ros2 launch fabric_sim_bringup full_sim.launch.py`, or directly:
    <isaac>/python.sh fabric_newton_isaacsim.py --headless --smoke-frames 260 \
        --cloth-start-z 0.85 --smoke-leader-velocity 0 0 -0.1 --smoke-output /tmp/smoke
"""
import os
# One BLAS thread, set before NumPy loads: the filter's small LAPACK calls
# stall on OpenBLAS thread hand-off (p95 140 ms with 24 threads, 3 ms with one).
BLAS_THREADS = os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Load NumPy, SciPy and the pure-NumPy filter modules BEFORE Kit starts: Kit
# sets OPENBLAS_NUM_THREADS=16 during start-up, and an OpenBLAS loaded after
# that keeps 15 worker threads spinning, which slowed the filter about 1.7x.
import numpy as np  # noqa: E402
from capsule_cbf.capsule import Capsule  # noqa: E402
from capsule_cbf.cloth_state import ClothState, particle_state_payload  # noqa: E402
from capsule_cbf.reduced_cloth import ReducedCloth  # noqa: E402
from capsule_cbf.sensitivity_cbf import SensitivityCbfController  # noqa: E402
from capsule_cbf.settings import load_settings  # noqa: E402

REPO = Path(__file__).resolve().parent
CORNER_NAMES = ('follower_left', 'follower_right', 'leader_left', 'leader_right')
# Large or per-point debug entries kept out of the JSON diagnostics.
DEBUG_ARRAYS = ('surfaces', 'rows', 'rhs', 'candidate_indices')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    a = p.add_argument
    a('--grid-x', type=int, default=56, help='Cloth cells along X.')
    a('--grid-y', type=int, default=56, help='Cloth cells along Y.')
    a('--cell-x', type=float, default=0.015, help='Cell size along X (m).')
    a('--cell-y', type=float, default=0.015, help='Cell size along Y (m).')
    a('--cloth-center-x', type=float, default=0.0, help='Initial cloth center X (m).')
    a('--cloth-center-y', type=float, default=0.0, help='Initial cloth center Y (m).')
    a('--cloth-start-z', type=float, default=1.05,
      help='Initial cloth height (m); leave room for start-up sag above the capsule.')
    a('--areal-density', type=float, default=0.2,
      help='Cloth mass per area (kg/m^2); woven fabric is about 0.1-0.4.')
    a('--particle-radius', type=float, default=0.008, help='Cloth particle radius (m).')
    a('--table-hx', type=float, default=0.22, help='Table half size X (m).')
    a('--table-hy', type=float, default=0.22, help='Table half size Y (m).')
    a('--table-hz', type=float, default=0.16, help='Table half size Z (m).')
    a('--gripper-size', type=int, default=3, help='Gripper patch side in particles (3 -> 3x3).')
    a('--gripper-inset', type=float, default=0.05,
      help='Before frame 0, move every gripper this far (m) toward the cloth center along x and y, '
           'so the cloth hangs with natural sag instead of being stretched flat; 0 keeps the corners.')
    a('--substeps', type=int, default=8, help='Solver substeps per 60 Hz frame.')
    a('--iterations', type=int, default=10, help='Style3D nonlinear (PD) iterations.')
    a('--linear-iterations', type=int, default=16, help='Style3D PCG iterations.')
    a('--state-publish-every', type=int, default=3, help='Publish /fabric/sim/state every N frames.')
    a('--tri-ke-weft', type=float, default=3.0e4, help='Weft stretch stiffness.')
    a('--tri-ke-warp', type=float, default=3.0e4, help='Warp stretch stiffness.')
    a('--tri-ke-shear', type=float, default=6.0e3, help='Shear stiffness.')
    a('--tri-ka', type=float, default=5.0e3, help='Area stiffness (stored; unused by Style3D, Newton 1.6).')
    a('--tri-kd', type=float, default=5.0e-5, help='Triangle damping (stored; unused by Style3D, Newton 1.6).')
    a('--edge-ke-weft', type=float, default=2.0e-3, help='Weft bending stiffness.')
    a('--edge-ke-warp', type=float, default=1.0e-3, help='Warp bending stiffness.')
    a('--edge-ke-shear', type=float, default=5.0e-4, help='Shear bending stiffness.')
    a('--edge-kd', type=float, default=1.0e-5, help='Bending damping (stored; unused by Style3D, Newton 1.6).')
    a('--contact-radius', type=float, default=0.01,
      help='Soft contact radius (stored; unused by the Newton 1.6 collision pipeline).')
    a('--contact-margin', type=float, default=0.012, help='Soft contact gap (m): detection distance added to the particle radius.')
    a('--contact-ke', type=float, default=4.0e3, help='Soft contact stiffness.')
    a('--contact-kd', type=float, default=1.0e-2, help='Soft contact damping.')
    a('--contact-mu', type=float, default=2.0, help='Soft contact friction.')
    a('--ros-frame-id', default='world', help='Frame id of published ROS 2 messages.')
    a('--cmd-timeout', type=float, default=1.0, help='Leader commands older than this (s) count as zero.')
    a('--disable-cuda-graph', action='store_true', help='Step the live cloth without CUDA graph capture.')
    a('--headless', action='store_true', help='Run Kit without a window.')
    a('--cbf-config', type=Path, default=REPO/'capsule_cbf'/'config.json',
      help='Sensitivity CBF settings (JSON, see capsule_cbf/settings.py).')
    a('--smoke-frames', type=int, default=0,
      help='Exit after this many frames and run end-of-run checks; 0 runs until the window closes.')
    a('--smoke-output', type=Path, help='Smoke mode: directory for diagnostics, USD and viewport PNG.')
    a('--smoke-leader-velocity', type=float, nargs=3, default=(0., 0., 0.), metavar=('VX', 'VY', 'VZ'),
      help='Smoke mode: leader velocity (m/s) applied from --smoke-start-frame instead of ROS.')
    a('--smoke-start-frame', type=int, default=60, help='Smoke mode: first frame of leader motion.')
    return p


# Arguments not recognized here are left for Kit.
ARGS, KIT_ARGS = build_parser().parse_known_args()
if ARGS.smoke_frames > 0:
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)
sys.argv = [sys.argv[0], *KIT_ARGS]

from isaacsim import SimulationApp  # noqa: E402

# Must be created before any Omniverse import.
simulation_app = SimulationApp({'headless': ARGS.headless})
os.environ['OPENBLAS_NUM_THREADS'] = BLAS_THREADS  # undo Kit's override for anything loaded later

import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt  # noqa: E402
import warp as wp  # noqa: E402

# Isaac Sim 6.2 moved isaacsim.core.utils to extsDeprecated; use Kit directly.
if not omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.ros2.bridge', True):
    raise RuntimeError('Cannot enable isaacsim.ros2.bridge')
simulation_app.update()

import rclpy  # noqa: E402
from geometry_msgs.msg import Pose, PoseArray, TwistStamped  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String  # noqa: E402
import newton  # noqa: E402
from newton.solvers import style3d  # noqa: E402

from capsule_cbf.newton_runtime import ClothRuntime  # noqa: E402


# --------------------------------------------------------------------- USD

def new_stage():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, '/World')
    UsdLux.DistantLight.Define(stage, Sdf.Path('/World/DistantLight')).CreateIntensityAttr(3000.0)
    ground = UsdGeom.Mesh.Define(stage, '/World/Ground')
    s = 2.5
    ground.CreatePointsAttr([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0), Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDoubleSidedAttr(True)
    ground.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(.45, .45, .45)])
    return stage


def usd_box(stage, path, center, half, color=(.2, .25, .35)):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xform.AddScaleOp().Set(Gf.Vec3f(*(2*float(h) for h in half)))


def usd_points(stage, path, points, color, width):
    prim = UsdGeom.Points.Define(stage, path)
    prim.CreatePointsAttr().Set(np.asarray(points, dtype=float).reshape(-1, 3).tolist())
    prim.CreateWidthsAttr().Set([width])
    prim.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    prim.CreateDisplayColorAttr().Set([color])
    return prim


def usd_lines(stage, path, lines, color, width):
    prim = UsdGeom.BasisCurves.Define(stage, path)
    prim.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    prim.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    prim.CreateCurveVertexCountsAttr().Set([len(line) for line in lines])
    prim.CreatePointsAttr().Set(np.concatenate(lines).tolist() if len(lines) else [])
    prim.CreateWidthsAttr().Set([width])
    prim.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    prim.CreateDisplayColorAttr().Set([color])
    return prim


def vec3f_array(points):
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32))


# --------------------------------------------------------------------- ROS

class FabricRosIo:
    """Leader commands in; applied follower commands, cloth state, timing and CBF debug out."""

    def __init__(self, frame_id, cmd_timeout):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('fabric_newton_isaacsim')
        self.frame_id, self.cmd_timeout = frame_id, float(cmd_timeout)
        self.leader_cmds = np.zeros((2, 3), dtype=np.float32)
        self.leader_stamps = [None, None]
        self.callbacks = 0  # leader messages handled, to drain the queue each frame
        for i, side in enumerate(('left', 'right')):
            self.node.create_subscription(TwistStamped, f'/fabric/leader_{side}/cmd_vel',
                                          lambda msg, i=i: self._leader_cb(i, msg), 10)
        twist = lambda topic: self.node.create_publisher(TwistStamped, topic, 10)  # noqa: E731
        self.pub_applied = [twist('/fabric/follower_left/applied_cmd_vel'),
                            twist('/fabric/follower_right/applied_cmd_vel')]
        # Same follower commands under end-effector names, for a downstream robot bridge.
        self.pub_robot = [twist('/robot_left/ee_cmd_vel'), twist('/robot_right/ee_cmd_vel')]
        self.pub_corners = self.node.create_publisher(PoseArray, '/fabric/sim/corners', 10)
        self.pub_state = self.node.create_publisher(PointCloud2, '/fabric/sim/state', 10)
        self.pub_timing = self.node.create_publisher(Float64MultiArray, '/fabric/sim/timing', 10)
        self.pub_debug = self.node.create_publisher(String, '/fabric/capsule_cbf/debug', 10)
        self.node.get_logger().info(
            'ROS 2 I/O ready: /fabric/leader_{left,right}/cmd_vel in; '
            '/fabric/follower_{left,right}/applied_cmd_vel, /fabric/sim/{corners,state,timing}, '
            '/fabric/capsule_cbf/debug out')

    def _leader_cb(self, i, msg):
        v = msg.twist.linear
        self.leader_cmds[i] = (v.x, v.y, v.z)
        self.leader_stamps[i] = time.monotonic()
        self.callbacks += 1

    def spin_once(self, max_callbacks=64):
        # rclpy.spin_once runs ONE callback. Two 30 Hz leader topics outpace a
        # ~30 fps frame, so one callback per frame left stale commands queued:
        # leaders kept moving after the key was released and the two leaders
        # briefly received different commands. Drain the queue without waiting.
        for _ in range(max_callbacks):
            handled = self.callbacks
            rclpy.spin_once(self.node, timeout_sec=0.0)
            if self.callbacks == handled:
                break

    def leader_commands(self):
        now, out = time.monotonic(), np.zeros((2, 3), dtype=np.float32)
        for i, stamp in enumerate(self.leader_stamps):
            if stamp is not None and now-stamp <= self.cmd_timeout:
                out[i] = self.leader_cmds[i]
        return out

    def _stamp(self, msg):
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        return msg

    def publish_followers(self, velocities):
        for pubs in (self.pub_applied, self.pub_robot):
            for pub, v in zip(pubs, velocities):
                msg = self._stamp(TwistStamped())
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, v)
                pub.publish(msg)

    def publish_corners(self, positions):
        msg = self._stamp(PoseArray())
        for p in positions:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, p)
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_corners.publish(msg)

    def publish_state(self, positions, velocities):
        names, points = particle_state_payload(positions, velocities)
        msg = self._stamp(PointCloud2())
        msg.height, msg.width = 1, int(points.shape[0])
        msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(names)]
        msg.is_bigendian = False
        msg.point_step = 4*len(names)
        msg.row_step = msg.point_step*msg.width
        msg.is_dense = bool(np.isfinite(points).all())
        msg.data = points.tobytes()
        self.pub_state.publish(msg)

    def publish_timing(self, sim_time, frame_ms, cbf_ms):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label='sim_time_s,frame_ms,fps,cbf_ms', size=4, stride=4)]
        msg.data = [float(sim_time), float(frame_ms), 1000.0/frame_ms if frame_ms > 0 else 0.0, float(cbf_ms)]
        self.pub_timing.publish(msg)

    def publish_debug(self, report):
        msg = String()
        msg.data = json.dumps(report)
        self.pub_debug.publish(msg)

    def shutdown(self):
        try:
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


# -------------------------------------------------------------- simulation

class FabricSimulation:
    fps = 60

    def __init__(self, args, settings):
        self.args, self.settings = args, settings
        self.frame_dt = 1.0/self.fps
        self.substeps = args.substeps
        self.frame = 0
        self.sim_time = 0.0
        self.ros = None
        self.particle_radius = args.particle_radius
        self.table_half = np.array([args.table_hx, args.table_hy, args.table_hz], dtype=float)
        self.stage = new_stage()
        usd_box(self.stage, '/World/Table', (0., 0., args.table_hz), self.table_half)
        self._build_model()
        self._build_controller()
        self._create_visuals()
        self.debug = None
        self.stats = dict(min_clearance=float('inf'), max_tension=0., max_correction=0., status={},
                          cbf_ms=[], frame_ms=[])
        self.last_log = 0.0
        self._print_summary()

    # ------------------------------------------------------------- model
    def _build_model(self):
        a = self.args
        nx, ny = a.grid_x, a.grid_y
        width, height = nx*a.cell_x, ny*a.cell_y
        # Style3D spreads the given per-particle mass uniformly over the area.
        self.mass_per_particle = a.areal_density*width*height/((nx+1)*(ny+1))
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        shape = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, ke=5.0e5, kd=1.0e1)
        builder.add_ground_plane(cfg=shape)
        builder.add_shape_box(body=-1, cfg=shape, xform=wp.transform(wp.vec3(0., 0., a.table_hz), wp.quat_identity()),
                              hx=a.table_hx, hy=a.table_hy, hz=a.table_hz)
        self.cloth_start = len(builder.particle_q)
        style3d.add_cloth_grid(
            builder, pos=wp.vec3(a.cloth_center_x-width/2, a.cloth_center_y-height/2, a.cloth_start_z),
            rot=wp.quat_identity(), vel=wp.vec3(), dim_x=nx, dim_y=ny, cell_x=a.cell_x, cell_y=a.cell_y,
            mass=self.mass_per_particle,
            tri_aniso_ke=wp.vec3(a.tri_ke_weft, a.tri_ke_warp, a.tri_ke_shear), tri_ka=a.tri_ka, tri_kd=a.tri_kd,
            edge_aniso_ke=wp.vec3(a.edge_ke_weft, a.edge_ke_warp, a.edge_ke_shear), edge_kd=a.edge_kd,
            particle_radius=self.particle_radius)
        self.cloth_count = len(builder.particle_q)-self.cloth_start
        rest = np.asarray(builder.particle_q[self.cloth_start:], dtype=float)
        self.corner_indices, self.patch, self.patch_corners, self.patch_offsets = self._gripper_patches(rest)
        # Prescribed gripper DOFs must not move inside Newton's nonlinear solve.
        for index in self.patch:
            builder.particle_flags[int(index)] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_radius = a.contact_radius
        self.model.soft_contact_margin = a.contact_margin
        self.model.soft_contact_ke = a.contact_ke
        self.model.soft_contact_kd = a.contact_kd
        self.model.soft_contact_mu = a.contact_mu
        # SolverStyle3D (Newton 1.6) precomputes its PD matrix in __init__.
        self.use_cuda_graph = not a.disable_cuda_graph
        self.motion = ClothRuntime(self.model, self.patch, self.patch_corners, self.patch_offsets,
                                   self.substeps, self.frame_dt/self.substeps, a.iterations,
                                   a.linear_iterations, capture=self.use_cuda_graph)
        corners = self.motion.state.particle_q.numpy()[self.corner_indices].astype(float)
        # Pre-roll (not counted as simulation frames): bring the grippers in and
        # let the cloth settle; these become the initial corner positions and the
        # CBF formation reference.
        self.initial_corners = self.motion.inset_corners(corners, a.gripper_inset, self.frame_dt)
        self.corner_targets = self.initial_corners.copy()

    def _gripper_patches(self, rest):
        """Square patches of gripper_size x gripper_size particles at each corner.

        Corner order: follower_left (min x, min y), follower_right (max x, min y),
        leader_left (min x, max y), leader_right (max x, max y).
        """
        a = self.args
        lo, hi = rest[:, :2].min(axis=0), rest[:, :2].max(axis=0)
        reach = (a.gripper_size-1)*np.array([a.cell_x, a.cell_y])+.5*np.array([a.cell_x, a.cell_y])
        tips = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]])
        corners, patch, ids, offsets = [], [], [], []
        for cid, tip in enumerate(tips):
            tip_local = int(np.argmin(np.linalg.norm(rest[:, :2]-tip, axis=1)))
            if np.linalg.norm(rest[tip_local, :2]-tip) > .25*min(a.cell_x, a.cell_y):
                raise RuntimeError(f'Cannot locate the {CORNER_NAMES[cid]} corner particle')
            members = np.flatnonzero(np.all(np.abs(rest[:, :2]-tip) <= reach, axis=1))
            corners.append(self.cloth_start+tip_local)
            patch.extend(self.cloth_start+members)
            ids.extend([cid]*len(members))
            offsets.extend(rest[members]-rest[tip_local])
        if len(set(patch)) != len(patch):
            raise RuntimeError('Gripper patches overlap; reduce --gripper-size')
        return (np.asarray(corners), np.asarray(patch, dtype=np.int32), np.asarray(ids, dtype=np.int32),
                np.asarray(offsets, dtype=np.float32))

    def _build_controller(self):
        cfg = self.settings
        self.capsule = Capsule([0., 0., self.args.table_hz], self.table_half, cfg.top_cap_mode,
                               cfg.top_cap_radius_scale)
        reduced = ReducedCloth(
            self.model, self.patch, self.patch_corners, self.patch_offsets, self.frame_dt,
            particle_start=self.cloth_start, particle_count=self.cloth_count,
            particle_radius=self.particle_radius, cells=cfg.prediction_grid_cells,
            substeps=cfg.model_substeps, stride=cfg.horizon_stride, plant_substeps=self.substeps,
            jacobian=cfg.model_jacobian)
        self.controller = SensitivityCbfController(self.capsule, reduced, self.initial_corners,
                                                   self.particle_radius, cfg, self.frame_dt)

    # ----------------------------------------------------------- visuals
    def _create_visuals(self):
        start = self.cloth_start
        mesh = UsdGeom.Mesh.Define(self.stage, '/World/Cloth')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(.75, .15, .15)])
        triangles = self.model.tri_indices.numpy()-start
        mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
        mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        mesh.CreatePointsAttr(vec3f_array(self.cloth_positions()))
        self.cloth_mesh = mesh
        self.corner_markers = []
        colors = ((.15, .35, 1.), (.15, .35, 1.), (.1, .75, .2), (.1, .75, .2))  # followers blue, leaders green
        for i, (name, color) in enumerate(zip(CORNER_NAMES, colors)):
            sphere = UsdGeom.Sphere.Define(self.stage, f'/World/Corners/{name}')
            sphere.CreateRadiusAttr(.016)
            sphere.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(*color)])
            op = UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp()
            op.Set(Gf.Vec3d(*map(float, self.corner_targets[i])))
            self.corner_markers.append(op)
        # Capsule envelope: USD only, never part of Newton's collision model.
        cfg, cap = self.settings, self.capsule
        lines = [cap.surface(np.linspace(0, 1, 81), [angle])
                 for angle in np.linspace(0, 2*np.pi, cfg.sectors, endpoint=False)]
        lines += [cap.surface([f], np.linspace(0, 2*np.pi, 97)) for f in np.linspace(0, 1, cfg.bands+1)[1:-1]]
        usd_lines(self.stage, '/World/CapsuleCBF/Boundary', lines, (.12, .5, .8), .0015)
        self._show_candidates(np.empty((0, 3)), np.empty((0, 3)))

    def _show_candidates(self, points, surfaces):
        """Clearance rows of the current frame (orange) and their capsule projections."""
        usd_points(self.stage, '/World/CapsuleCBF/Candidates', points, (1., .65, .05), .012)
        usd_lines(self.stage, '/World/CapsuleCBF/DistanceSegments',
                  [np.stack((s, p)) for s, p in zip(surfaces, points)], (1., .65, .05), .002)

    def _update_visuals(self, positions):
        self.cloth_mesh.GetPointsAttr().Set(vec3f_array(positions))
        for op, p in zip(self.corner_markers, self.corner_positions()):
            op.Set(Gf.Vec3d(*map(float, p)))
        # Debug geometry edits force a stage sync; 5 Hz is enough.
        if self.frame % 12 == 0 and self.debug is not None:
            idx = np.asarray(self.debug.get('candidate_indices', []), dtype=int)
            self._show_candidates(positions[idx], self.debug.get('surfaces', np.empty((0, 3))))

    # -------------------------------------------------------------- state
    def cloth_state(self):
        return ClothState.from_newton(self.motion.state, self.cloth_start, self.cloth_count)

    def cloth_positions(self):
        return self.motion.state.particle_q.numpy()[self.cloth_start:self.cloth_start+self.cloth_count]

    def corner_positions(self):
        return self.motion.state.particle_q.numpy()[self.corner_indices].astype(float)

    # --------------------------------------------------------------- step
    def start_ros(self):
        if self.ros is None:
            self.ros = FabricRosIo(self.args.ros_frame_id, self.args.cmd_timeout)

    def _leader_commands(self):
        if self.args.smoke_frames > 0:
            moving = self.frame >= self.args.smoke_start_frame
            return np.tile(np.asarray(self.args.smoke_leader_velocity, dtype=float) if moving else np.zeros(3), (2, 1))
        return self.ros.leader_commands().astype(float) if self.ros is not None else np.zeros((2, 3))

    def _follower_commands(self, state, leaders):
        cfg = self.settings
        if self.frame < cfg.warmup_frames:
            # The spawned flat sheet is slack; the affine model is linearized
            # about hanging, tensioned cloth (SENSITIVITY_CBF_REPORT.md, step 3).
            d = self.capsule.distances(state.positions, self.particle_radius)[0]
            return leaders.copy(), dict(status='warmup', mode=cfg.controller, min_clearance=float(d.min()),
                                        h_min=float(d.min()-cfg.margin), correction=0., controller_ms=0.,
                                        candidate_indices=[], surfaces=np.empty((0, 3)))
        return self.controller.update(state, self.corner_positions(), leaders, self.frame_dt)

    def step(self):
        started = time.perf_counter()
        if self.ros is not None:
            self.ros.spin_once()
        leaders = self._leader_commands()
        state = self.cloth_state()
        followers, debug = self._follower_commands(state, leaders)
        debug.update(frame=self.frame, sim_time=self.sim_time, warmup=self.frame < self.settings.warmup_frames)
        self.debug = debug
        velocities = np.vstack((followers, leaders))
        self.motion.advance(self.corner_targets, velocities)
        self.corner_targets = self.corner_targets+velocities*self.frame_dt
        self.sim_time += self.frame_dt
        positions = self.cloth_positions()
        self._update_visuals(positions)
        frame_ms = 1000*(time.perf_counter()-started)
        self._record(debug, frame_ms)
        if self.ros is not None:
            self.ros.publish_followers(followers)
            self.ros.publish_corners(self.corner_positions())
            if self.frame % max(1, self.args.state_publish_every) == 0:
                after = self.cloth_state()
                self.ros.publish_state(after.positions, after.velocities)
            self.ros.publish_timing(self.sim_time, frame_ms, debug.get('controller_ms', 0.))
            self.ros.publish_debug({k: v for k, v in debug.items() if k not in DEBUG_ARRAYS})
        self.frame += 1

    def _record(self, debug, frame_ms):
        s = self.stats
        s['min_clearance'] = min(s['min_clearance'], debug['min_clearance'])
        s['max_tension'] = max(s['max_tension'], max(debug.get('tension', [0.])))
        s['max_correction'] = max(s['max_correction'], debug['correction'])
        s['status'][debug['status']] = s['status'].get(debug['status'], 0)+1
        if not debug['warmup']:
            s['cbf_ms'].append(debug['controller_ms'])
        s['frame_ms'].append(frame_ms)
        now = time.perf_counter()
        if now-self.last_log >= 2.0:
            self.last_log = now
            tension = max(debug.get('tension', [float('nan')]))
            print(f"[fabric_sim] frame {self.frame} {debug['status']}: clearance {debug['min_clearance']:.4f} m, "
                  f"max tension {tension:.1f} N, correction {debug['correction']:.3f} m/s, "
                  f"cbf {debug['controller_ms']:.1f} ms, frame {frame_ms:.1f} ms", flush=True)

    def summary(self):
        s = self.stats
        pct = lambda x, q: float(np.percentile(x, q)) if len(x) else None  # noqa: E731
        return dict(frames=self.frame, controller=self.settings.controller,
                    run_min_clearance=s['min_clearance'], run_max_tension=s['max_tension'],
                    run_max_correction=s['max_correction'], status_counts=s['status'],
                    cbf_ms_p50=pct(s['cbf_ms'], 50), cbf_ms_p95=pct(s['cbf_ms'], 95),
                    cbf_ms_max=max(s['cbf_ms']) if s['cbf_ms'] else None,
                    frame_ms_p50=pct(s['frame_ms'], 50),
                    final={k: v for k, v in self.debug.items() if k not in DEBUG_ARRAYS})

    def check_final_state(self):
        """End-of-smoke sanity: cloth above ground, grippers exactly on their targets."""
        q = self.motion.state.particle_q.numpy()
        if q[self.cloth_start:, 2].min() < -0.05:
            raise ValueError('Cloth particles fell through the ground plane')
        error = np.linalg.norm(q[self.corner_indices]-self.corner_targets, axis=1).max()
        if error > 1e-4:
            raise ValueError(f'Gripper corners drifted from their targets by {error:.3e} m')

    def write_smoke_output(self, output):
        output.mkdir(parents=True, exist_ok=True)
        self.stage.Export(str(output/'scene.usda'))
        (output/'diagnostics.json').write_text(json.dumps(self.summary(), indent=2))
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            import asyncio
            viewport = get_active_viewport()
            camera = ViewportCameraState('/OmniverseKit_Persp', viewport)
            camera.set_position_world(Gf.Vec3d(1.5, -1.8, 1.25), True)
            camera.set_target_world(Gf.Vec3d(0., 0., .4), True)
            for _ in range(10):
                simulation_app.update()
            capture = capture_viewport_to_file(viewport, str(output/'viewport.png'))
            done = asyncio.ensure_future(capture.wait_for_result())
            for _ in range(120):
                simulation_app.update()
                if done.done():
                    done.result()
                    break
            if not done.done():
                raise RuntimeError('viewport capture did not finish')
        except Exception as error:  # a missing viewport (e.g. headless) must not fail the run
            print(f'[fabric_sim] viewport capture skipped: {error}')

    def _print_summary(self):
        a, cfg, info, cap = self.args, self.settings, self.controller.red.info, self.capsule
        print(f'[fabric_sim] cloth {a.grid_x}x{a.grid_y} cells ({self.cloth_count} particles), '
              f'areal density {a.areal_density} kg/m^2 -> {self.mass_per_particle:.3e} kg/particle, '
              f'center ({a.cloth_center_x:.3f}, {a.cloth_center_y:.3f}, {a.cloth_start_z:.3f}) m')
        print(f'[fabric_sim] solver: {self.substeps} substeps, {a.iterations} PD / {a.linear_iterations} PCG '
              f'iterations, CUDA graph {"on" if self.motion.graph is not None else "off"}; '
              f'gripper patches {a.gripper_size}x{a.gripper_size} ({len(self.patch)} pinned particles), '
              f'inset {a.gripper_inset:.3f} m; initial corners (x, y) '
              f'{np.round(self.initial_corners[:, :2], 3).tolist()}')
        print(f'[fabric_sim] CBF controller={cfg.controller}, capsule radius {self.capsule.radius:.4f} m, '
              f'top={cfg.top_cap_mode} (sphere radius {cap.top_sphere_radius:.3f} m, '
              f'apex z {cap.center[2]+cap.top_center+cap.top_sphere_radius:.3f} m), '
              f'margin {cfg.margin} m, tension limit {cfg.tension_limit} N, '
              f'strain={cfg.strain_enabled}, warm-up {cfg.warmup_frames} frames; prediction model: {info}')
        if KIT_ARGS:
            print(f'[fabric_sim] arguments passed to Kit: {KIT_ARGS}')

    def shutdown(self):
        if self.ros is not None:
            self.ros.shutdown()
            self.ros = None


def main():
    sim, exit_code = None, 1
    try:
        sim = FabricSimulation(ARGS, load_settings(ARGS.cbf_config))
        # Let Kit render and the ROS bridge settle before the first step mutates USD.
        simulation_app.update()
        sim.start_ros()
        while simulation_app.is_running():
            sim.step()
            simulation_app.update()
            if 0 < ARGS.smoke_frames <= sim.frame:
                sim.check_final_state()
                if ARGS.smoke_output:
                    sim.write_smoke_output(ARGS.smoke_output)
                print('[fabric_sim] smoke run completed:', json.dumps(
                    {k: v for k, v in sim.summary().items() if k != 'final'}), flush=True)
                break
        exit_code = 0
    except KeyboardInterrupt:  # Ctrl+C or launch shutdown
        exit_code = 0
    except Exception:
        # Kit's fast shutdown exits the process; print the error first.
        traceback.print_exc()
    finally:
        if sim is not None:
            sim.shutdown()
        simulation_app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()

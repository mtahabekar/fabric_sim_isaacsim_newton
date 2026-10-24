#!/usr/bin/env python3
"""Synchronous capsule CBF with configurable coarse or full Newton prediction."""
import argparse
import json
import sys
import time
from pathlib import Path

# Consume new arguments before the legacy module starts Kit and parses its flags.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--capsule-config', default=str(Path(__file__).parent / 'capsule_cbf/config.json'))
parser.add_argument('--capsule-smoke-frames', type=int, default=0,
                    help='Exit after this many frames; zero runs interactively.')
parser.add_argument('--capsule-smoke-output', type=Path,
                    help='Optional directory for final smoke diagnostics, USD, and viewport PNG.')
parser.add_argument('--capsule-smoke-leader-z', type=float, default=0.,
                    help='In smoke mode only, command both leaders along Z after 60 settling frames.')
options, remaining = parser.parse_known_args()
if options.capsule_smoke_frames > 0:
    import faulthandler
    faulthandler.dump_traceback_later(90, repeat=True)
sys.argv = [sys.argv[0], *remaining]
if not any(arg.split('=')[0] == '--cloth-start-z' for arg in remaining):
    sys.argv.extend(['--cloth-start-z', '1.05'])
with open(options.capsule_config, encoding='utf-8') as stream:
    config = json.load(stream)
from capsule_cbf.core import Capsule, CapsuleController, Settings
settings = Settings(**config)
import fabric_newton_isaacsim_standalone as legacy
import numpy as np
from pxr import UsdGeom
from std_msgs.msg import String
from capsule_cbf.newton_model import CurrentNewtonModelMixin
from capsule_cbf.coarse_prediction import create_predictor
from capsule_cbf.predictive import PredictiveController
from capsule_cbf.ray_basins import RayBasinController
from capsule_cbf.newton_runtime import CapsuleRayDetector
from capsule_cbf.strain import TriangleStrain
from capsule_cbf.material_audit import audit_style3d


class CapsuleFabricSimulation(CurrentNewtonModelMixin, legacy.Style3DIsaacStandalone):
    def __init__(self, args):
        args.external_cbf_enabled = False
        # The capsule filter owns optional speed bounds; do not inherit a
        # separate cap from the legacy simulator arguments.
        args.follower_command_max_speed = settings.max_speed
        super().__init__(args)
        capsule = Capsule([0., 0., args.table_hz], self.table_half_extents,
                          settings.sectors, settings.bands, settings.top_cap_mode)
        self.capsule_controller = CapsuleController(capsule, self.get_cloth_particles(),
                                                    self.initial_corner_positions,
                                                    self.cloth_particle_radius, settings)
        # Hard-pinned patches have exact corner influence, unlike free cloth.
        for idx, corner in zip(self.gripper_flat_indices, self.gripper_flat_corner_ids):
            local = idx-self.cloth_particle_start
            self.capsule_controller.weights[local] = np.eye(4)[corner]
        self.predictor = None
        self.predicted_points = None
        if settings.predictive_enabled:
            triangles = self.model.tri_indices.numpy()-self.cloth_particle_start
            strain = TriangleStrain(triangles, self.model.tri_poses.numpy(),
                                    settings.weft_strain_limit, settings.warp_strain_limit,
                                    settings.shear_limit)
            self.predictive_controller = PredictiveController(
                capsule, strain, self.initial_corner_positions, self.cloth_particle_radius,
                settings, self.sim_substeps)
            self.predictor = create_predictor(
                self.model, self.gripper_flat_indices, self.gripper_flat_corner_ids,
                self.gripper_flat_offsets, self.sim_substeps, self.sim_dt,
                self.iterations, self.linear_iterations, capsule,
                self.cloth_particle_radius, settings,
                particle_start=self.cloth_particle_start,
                particle_count=self.cloth_particle_end-self.cloth_particle_start,
                capture=self.use_cuda_graph)
            self.predictor.progress_callback = legacy.simulation_app.update
            print('[capsule CBF] prediction resolution:', getattr(self.predictor, 'resolution_info',
                  {'prediction_model': 'full', 'prediction_particles': self.model.particle_count}))
            if settings.ray_basins_enabled:
                detector = CapsuleRayDetector(self.model, self.state_0, capsule)
                self.predictive_controller = RayBasinController(
                    capsule, strain, self.initial_corner_positions, self.cloth_particle_radius,
                    settings, self.sim_substeps, detector)
        self.material_audit = audit_style3d()
        print('[capsule material audit] tri_ka, tri_kd, edge_kd, and legacy '
              'contact_radius are not used by this installed Style3D path; '
              'strain limits are independent, uncalibrated policy values.')
        self.capsule_debug = None
        self.smoke_min_clearance = float('inf')
        self.smoke_recovery_frames = 0
        self.smoke_max_correction = 0.
        self.capsule_debug_pub = None
        self.last_capsule_log_time = 0.
        self._capsule_points('Cells', capsule.cells, (.2, .7, 1.), .005)
        # Wireframe meridians and rings; no physics/collision schemas or Newton shape.
        lines = []
        for angle in np.linspace(0, 2*np.pi, settings.sectors, endpoint=False):
            lines.append(capsule.surface(np.linspace(0, 1, 81), [angle]))
        for fraction in np.linspace(0, 1, settings.bands+1)[1:-1]:
            lines.append(capsule.surface([fraction], np.linspace(0, 2*np.pi, 97)))
        self._capsule_lines('Boundary', lines, (.12, .5, .8), .0015)
        self._capsule_points('Detections', [], (1., .65, .05), .012)
        self._capsule_points('BasinSeeds', [], (1., .1, .2), .018)
        self._capsule_lines('DistanceSegments', [], (1., .65, .05), .002)
        print(f'[capsule CBF] synchronous filter; radius={capsule.radius:.4f} m, '
              f'cells={len(capsule.cells)}, predictive={settings.predictive_enabled}, '
              f'ray_basins={settings.ray_basins_enabled}, strain={settings.strain_enabled}, '
              f'sensitivity={settings.sensitivity_method}/{settings.sensitivity_backend}')
        print(f'[capsule CBF] top={capsule.top_cap_mode}, '
              f'top sphere radius={capsule.top_sphere_radius:.4f} m')

    def _capsule_points(self, name, points, color, width):
        prim = UsdGeom.Points.Define(self.stage, '/World/CapsuleCBF/'+name)
        prim.CreatePointsAttr().Set(np.asarray(points).reshape(-1, 3).tolist())
        prim.CreateWidthsAttr().Set([width])
        prim.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        prim.CreateDisplayColorAttr().Set([color])
        return prim

    def _capsule_lines(self, name, lines, color, width):
        prim = UsdGeom.BasisCurves.Define(self.stage, '/World/CapsuleCBF/'+name)
        prim.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        prim.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
        prim.CreateCurveVertexCountsAttr().Set([len(line) for line in lines])
        prim.CreatePointsAttr().Set(np.concatenate(lines).tolist() if len(lines) else [])
        prim.CreateWidthsAttr().Set([width])
        prim.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        prim.CreateDisplayColorAttr().Set([color])
        return prim

    def _compute_follower_applied_velocities(self, leader_cmd_vels):
        if self.predictor is None:
            points = self.get_cloth_particles()
            command, debug = self.capsule_controller.update(
                points, self._get_actual_corner_positions(), leader_cmd_vels, self.frame_dt)
        else:
            prediction_started = time.perf_counter()
            self.predictor.begin(self.state_0, self.corner_target_positions)
            # The CBF and predictor see the same full-particle q/v snapshot.
            cloth_state = self.predictor.current_cloth_state()
            points = cloth_state.positions
            command, debug = self.predictive_controller.update(
                cloth_state, self._get_actual_corner_positions(), leader_cmd_vels,
                self.frame_dt, self.predictor)
            if self.predicted_points is not None:
                error = points-self.predicted_points
                debug['prediction_position_rmse'] = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                debug['prediction_position_max_error'] = float(np.linalg.norm(error, axis=1).max())
            # The returned command can be an earlier best recovery candidate;
            # the runtime's final rollout need not correspond to it. Record
            # the expectation only when the final validated candidate won.
            self.predicted_points = (self.predictor.final_points()
                                     if debug['status'] == 'feasible' and debug['rollout_count'] else None)
            debug['prediction_total_ms'] = 1000*(time.perf_counter()-prediction_started)
            debug['last_sensitivity_timings'] = self.predictor.last_sensitivity_timings
            debug.update(getattr(self.predictor, 'resolution_info', {'prediction_model': 'full'}))
        self.capsule_debug = debug
        self.smoke_min_clearance = min(self.smoke_min_clearance, debug['min_clearance'])
        self.smoke_recovery_frames += int(debug['status'] != 'feasible')
        self.smoke_max_correction = max(self.smoke_max_correction, debug['correction'])
        self.last_applied_follower_vels = command.astype(np.float32)
        detected = debug.get('detected_points', points[debug['detected_indices']])
        basins = debug.get('basin_points', points[debug['basin_indices']])
        self._capsule_points('Detections', detected, (1., .65, .05), .012)
        self._capsule_points('BasinSeeds', basins, (1., .1, .2), .018)
        lines = [np.stack((surface, p))
                 for surface, p in zip(debug['surfaces'], detected)]
        self._capsule_lines('DistanceSegments', lines, (1., .65, .05), .002)
        if self.ros is not None:
            if self.capsule_debug_pub is None:
                self.capsule_debug_pub = self.ros.node.create_publisher(String, '/fabric/capsule_cbf/debug', 10)
            report = {k: v for k, v in debug.items()
                      if k not in ('surfaces', 'rows', 'rhs', 'detected_indices', 'basin_indices', 'detected_points', 'basin_points')}
            msg = String()
            msg.data = json.dumps(report)
            self.capsule_debug_pub.publish(msg)
            now = time.perf_counter()
            if now-self.last_capsule_log_time >= 2.:
                self.ros.node.get_logger().info('[capsule CBF] '+msg.data)
                self.last_capsule_log_time = now
        return self.last_applied_follower_vels

    def _get_leader_velocity_commands(self):
        if options.capsule_smoke_frames > 0:
            velocity = options.capsule_smoke_leader_z if self._frame_index >= 60 else 0.
            return np.array([[0., 0., velocity], [0., 0., velocity]], dtype=np.float32)
        return super()._get_leader_velocity_commands()


if __name__ == '__main__':
    sim = None
    try:
        sim = CapsuleFabricSimulation(legacy.parse_args())
        legacy.simulation_app.update()
        sim.initialize_ros_io()
        while legacy.simulation_app.is_running():
            sim.step()
            legacy.simulation_app.update()
            if options.capsule_smoke_frames > 0 and sim._frame_index >= options.capsule_smoke_frames:
                sim.test_final()
                if options.capsule_smoke_output:
                    output = options.capsule_smoke_output
                    output.mkdir(parents=True, exist_ok=True)
                    sim.stage.Export(str(output/'capsule_scene.usda'))
                    report = {k: v for k, v in sim.capsule_debug.items()
                              if k not in ('surfaces', 'rows', 'rhs', 'detected_points', 'basin_points')}
                    report.update(run_min_clearance=sim.smoke_min_clearance,
                                  recovery_frames=sim.smoke_recovery_frames,
                                  run_max_correction=sim.smoke_max_correction)
                    (output/'diagnostics.json').write_text(json.dumps(report, indent=2))
                    (output/'material_audit.json').write_text(json.dumps(sim.material_audit, indent=2))
                    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                    from isaacsim.core.utils.viewports import set_camera_view
                    set_camera_view(eye=np.array([1.5, -1.8, 1.25]),
                                    target=np.array([0., 0., .4]))
                    for _ in range(10):
                        legacy.simulation_app.update()
                    capture = capture_viewport_to_file(get_active_viewport(), str(output/'viewport.png'))
                    import asyncio
                    done = asyncio.ensure_future(capture.wait_for_result())
                    for _ in range(120):
                        legacy.simulation_app.update()
                        if done.done():
                            done.result()
                            break
                    if not done.done():
                        raise RuntimeError('Viewport capture did not finish')
                print('[capsule CBF] smoke run completed')
                break
    except Exception:
        # Kit fast shutdown can exit before Python prints a pending exception.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if sim is not None:
            sim.shutdown()
        legacy.simulation_app.close()

#!/usr/bin/env python3
"""Separate synchronous capsule CBF entry point; legacy simulator is unchanged."""
import argparse
import json
import sys
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


class CapsuleFabricSimulation(CurrentNewtonModelMixin, legacy.Style3DIsaacStandalone):
    def __init__(self, args):
        args.external_cbf_enabled = False
        super().__init__(args)
        if args.follower_command_max_speed > 0:
            settings.max_speed = min(settings.max_speed, args.follower_command_max_speed)
        capsule = Capsule([0., 0., args.table_hz], self.table_half_extents,
                          settings.sectors, settings.bands)
        self.capsule_controller = CapsuleController(capsule, self.get_cloth_particles(),
                                                    self.initial_corner_positions,
                                                    self.cloth_particle_radius, settings)
        # Hard-pinned patches have exact corner influence, unlike free cloth.
        for idx, corner in zip(self.gripper_flat_indices, self.gripper_flat_corner_ids):
            local = idx-self.cloth_particle_start
            self.capsule_controller.weights[local] = np.eye(4)[corner]
        self.capsule_debug = None
        self.smoke_min_clearance = float('inf')
        self.smoke_recovery_frames = 0
        self.smoke_max_correction = 0.
        self.capsule_debug_pub = None
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
              f'cells={len(capsule.cells)}, cell_guards={settings.cell_guards}')

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
        points = self.get_cloth_particles()
        command, debug = self.capsule_controller.update(
            points, self._get_actual_corner_positions(), leader_cmd_vels, self.frame_dt)
        self.capsule_debug = debug
        self.smoke_min_clearance = min(self.smoke_min_clearance, debug['min_clearance'])
        self.smoke_recovery_frames += int(debug['status'] != 'feasible')
        self.smoke_max_correction = max(self.smoke_max_correction, debug['correction'])
        self.last_applied_follower_vels = command.astype(np.float32)
        self._capsule_points('Detections', points[debug['detected_indices']], (1., .65, .05), .012)
        self._capsule_points('BasinSeeds', points[debug['basin_indices']], (1., .1, .2), .018)
        lines = [np.stack((surface, points[i]))
                 for surface, i in zip(debug['surfaces'], debug['detected_indices'])]
        self._capsule_lines('DistanceSegments', lines, (1., .65, .05), .002)
        if self.ros is not None:
            if self.capsule_debug_pub is None:
                self.capsule_debug_pub = self.ros.node.create_publisher(String, '/fabric/capsule_cbf/debug', 10)
            report = {k: v for k, v in debug.items()
                      if k not in ('surfaces', 'rows', 'rhs', 'detected_indices', 'basin_indices')}
            msg = String()
            msg.data = json.dumps(report)
            self.capsule_debug_pub.publish(msg)
            if self._frame_index % 60 == 0:
                self.ros.node.get_logger().info('[capsule CBF] '+msg.data)
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
                              if k not in ('surfaces', 'rows', 'rhs')}
                    report.update(run_min_clearance=sim.smoke_min_clearance,
                                  recovery_frames=sim.smoke_recovery_frames,
                                  run_max_correction=sim.smoke_max_correction)
                    (output/'diagnostics.json').write_text(json.dumps(report, indent=2))
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

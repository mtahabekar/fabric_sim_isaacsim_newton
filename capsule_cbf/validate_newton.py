"""Headless Newton validation/benchmark without Isaac Kit or ROS.

Run with Isaac Python and Newton/Warp extension directories on PYTHONPATH.
JSON reports include sensitivity error, nonlinear checks, and stage timings.
"""
import argparse
import json
from pathlib import Path
import time
import faulthandler
import numpy as np
import warp as wp
import newton
from newton.solvers import style3d
from capsule_cbf.core import Capsule, Settings
from capsule_cbf.strain import TriangleStrain
from capsule_cbf.newton_runtime import ClothRuntime, NewtonPredictor
from capsule_cbf.predictive import PredictiveController, six_direction_sensitivity
from capsule_cbf.ray_basins import RayBasinController
from capsule_cbf.newton_runtime import CapsuleRayDetector
from capsule_cbf.material_audit import audit_style3d


def build_fixture(grid, substeps, device, cfg, cloth_z=1.05):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(b)
    shape = newton.ModelBuilder.ShapeConfig(density=0., mu=.8, ke=5.e5, kd=10.)
    b.add_ground_plane(cfg=shape)
    b.add_shape_box(body=-1, cfg=shape, xform=wp.transform(wp.vec3(0, 0, .16), wp.quat_identity()),
                    hx=.22, hy=.22, hz=.16)
    style3d.add_cloth_grid(b, pos=wp.vec3(-.42, -.42, cloth_z), rot=wp.quat_identity(), vel=wp.vec3(),
                          dim_x=grid, dim_y=grid, cell_x=.84/grid, cell_y=.84/grid, mass=.003,
                          tri_aniso_ke=wp.vec3(3.e4, 3.e4, 6.e3), tri_ka=5.e3, tri_kd=5.e-5,
                          edge_aniso_ke=wp.vec3(.002, .001, .0005), edge_kd=1.e-5, particle_radius=.008)
    rest = np.asarray(b.particle_q, dtype=np.float32)
    wanted = np.array([[-.42, -.42], [.42, -.42], [-.42, .42], [.42, .42]])
    corners = np.array([np.argmin(np.linalg.norm(rest[:, :2]-p, axis=1)) for p in wanted])
    patch, ids, offsets = [], [], []
    reach = min(3, max(1, grid//4))
    for cid, p in enumerate(wanted):
        selected = np.flatnonzero(np.all(abs(rest[:, :2]-p) <= (reach-1)*.84/grid+1e-5, axis=1))
        for index in selected:
            patch.append(index)
            ids.append(cid)
            offsets.append(rest[index]-rest[corners[cid]])
            b.particle_flags[int(index)] &= ~int(newton.ParticleFlags.ACTIVE)
    model = b.finalize(device=device)
    model.set_gravity((0, 0, -9.81))
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 4000., .01, 2.
    model.soft_contact_margin = .012
    cap = Capsule([0, 0, .16], [.22, .22, .16])
    live = ClothRuntime(model, patch, ids, offsets, substeps, 1/60/substeps, 10, 16)
    predictor = NewtonPredictor(model, patch, ids, offsets, substeps, 1/60/substeps, 10, 16,
                                cap, .008, cfg)
    strain = TriangleStrain(model.tri_indices.numpy(), model.tri_poses.numpy(),
                            cfg.weft_strain_limit, cfg.warp_strain_limit, cfg.shear_limit)
    controller = PredictiveController(cap, strain, rest[corners], .008, cfg, substeps)
    if cfg.ray_basins_enabled:
        controller = RayBasinController(cap, strain, rest[corners], .008, cfg, substeps,
                                        CapsuleRayDetector(model, live.state, cap))
    return model, live, predictor, controller, corners, rest[corners]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--grid', type=int, default=8)
    p.add_argument('--substeps', type=int, default=8)
    p.add_argument('--frames', type=int, default=12)
    p.add_argument('--settle-frames', type=int, default=4)
    p.add_argument('--leader-z', type=float, default=-.05)
    p.add_argument('--cloth-z', type=float, default=1.05)
    p.add_argument('--output', type=Path, default=Path('/tmp/capsule-newton-validation.json'))
    args = p.parse_args()
    faulthandler.dump_traceback_later(30, repeat=True)
    if args.grid < 4 or args.frames < 1 or args.substeps < 1:
        p.error('grid >= 4, frames/substeps >= 1 required')
    wp.config.kernel_cache_dir = '/tmp/capsule-cbf-warp-cache'
    wp.init()
    cfg = Settings()
    with wp.ScopedDevice(args.device):
        model, live, pred, ctrl, corners, starts = build_fixture(args.grid, args.substeps, args.device, cfg,
                                                              args.cloth_z)
        u, leaders = np.zeros((2, 3)), np.zeros((2, 3))
        before = live.state.particle_q.numpy().copy()
        pred.begin(live.state, starts)
        base = pred(u, leaders)
        jac = six_direction_sensitivity(pred, u, leaders, cfg.sensitivity_step, base)
        central = []
        for d in np.eye(6):
            central.append((pred(u+.01*d.reshape(2, 3), leaders)
                            -pred(u-.01*d.reshape(2, 3), leaders))/.02)
        central = np.stack(central, axis=-1)
        direction = np.array([.7, -.3, .5, -.2, .6, -.4])
        holdout = pred(u+.01*direction.reshape(2, 3), leaders)
        estimated = base+.01*(jac@direction)
        isolation = bool(np.array_equal(before, live.state.particle_q.numpy()))
        records = []
        progress = args.output.with_suffix('.progress.jsonl')
        progress.write_text('')
        for frame in range(args.frames):
            leaders = np.zeros((2, 3))
            if frame >= args.settle_frames:
                leaders[:, 2] = args.leader_z
            points = live.state.particle_q.numpy()
            pred.begin(live.state, starts)
            begin = time.perf_counter()
            command, debug = ctrl.update(points, points[corners], leaders, 1/60, pred)
            control_ms = (time.perf_counter()-begin)*1000
            # Independent confirmation of the command actually returned,
            # including an earlier best candidate in infeasible recovery.
            predicted = pred(command, leaders)
            predicted_points = pred.final_points()
            begin = time.perf_counter()
            live.advance(starts, np.vstack((command, leaders)))
            wp.synchronize_device(args.device)
            physics_ms = (time.perf_counter()-begin)*1000
            actual = live.state.particle_q.numpy()
            debug = {k: v for k, v in debug.items() if k not in
                     ('rows', 'rhs', 'surfaces', 'detected_indices', 'basin_indices', 'detected_points', 'basin_points')}
            debug.update(frame=frame, control_ms=control_ms, physics_ms=physics_ms,
                         control_plus_physics_ms=control_ms+physics_ms,
                         live_prediction_max_error=float(np.linalg.norm(actual-predicted_points, axis=1).max()),
                         actual_min_clearance=float(ctrl.capsule.distances(actual, .008)[0].min()),
                         actual_min_strain_barrier=float(ctrl.strain.barriers(actual).min()))
            records.append(debug)
            with progress.open('a') as stream:
                stream.write(json.dumps(debug)+'\n')
            starts = starts+np.vstack((command, leaders)).astype(np.float32)/60
        report = dict(device=args.device, grid=args.grid, cloth_z=args.cloth_z,
                      particles=model.particle_count,
                      triangles=model.tri_count, substeps=args.substeps,
                      predictor_isolation=isolation,
                      sensitivity_max_central_error=float(abs(jac-central).max()),
                      sensitivity_heldout_barrier_error=float(abs(holdout-estimated).max()),
                      frames=records, material_audit=audit_style3d(),
                      timing_scope='Headless control + physics; excludes ROS/rendering and independent validation rollout.',
                      policy_limits=('Strain constraints disabled; clearance checked only at selected ray basin hits.'
                                     if cfg.ray_basins_enabled else
                                     'Configured material strain constraints are independent policy limits.'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({k: v for k, v in report.items() if k not in ('frames', 'material_audit')}, indent=2))
    print('Feasible frames:', sum(r['status'] == 'feasible' for r in records), '/', len(records))
    print('Max live/predicted position error:', max(r['live_prediction_max_error'] for r in records))
    print('Median / p95 control+physics ms:', np.percentile([r['control_plus_physics_ms'] for r in records], [50, 95]))
    print('Report:', args.output)


if __name__ == '__main__':
    main()

"""Closed-loop scenarios on the headless Newton plant: safety, stretch and timing.

Controllers: nominal (no filter), ray_basins (legacy predictive filter) and
sensitivity (this work). Scenarios start from a settled, safe cloth. Run with a
Newton 1.2.x environment; see SENSITIVITY_CBF_REPORT.md.
"""
import os
# Tiny LAPACK calls stall on OpenBLAS thread hand-off; one thread is faster.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from capsule_cbf.cloth_state import ClothState
from capsule_cbf.core import Settings
from capsule_cbf.scenario import FRAME_DT, build_plant
from capsule_cbf.strain import TriangleStrain

SCENARIOS = {
    # name: (leader velocity while moving, extra follower task velocity,
    #        motion frames, hold frames). The task velocity models a planner or
    #        teleoperation command that is unsafe for the fabric.
    'descend': (np.array([[0., 0., -.10], [0., 0., -.10]]), np.zeros((2, 3)), 200, 60),
    'pull': (np.zeros((2, 3)), np.array([[0., -.05, 0.], [0., -.05, 0.]]), 200, 60),
    'descend_pull': (np.array([[0., 0., -.10], [0., 0., -.10]]),
                     np.array([[0., -.03, 0.], [0., -.03, 0.]]), 200, 60),
}


def make_controller(name, plant, cfg):
    if name == 'nominal':
        return None
    if name == 'sensitivity':
        from capsule_cbf.reduced_cloth import ReducedCloth
        from capsule_cbf.sensitivity_cbf import SensitivityCbfController
        red = ReducedCloth(plant.model, plant.patch, plant.corner_ids, plant.offsets, FRAME_DT,
                           cells=cfg.prediction_grid_cells, substeps=cfg.model_substeps,
                           stride=cfg.horizon_stride, jacobian=cfg.model_jacobian)
        return SensitivityCbfController(plant.capsule, red, plant.rest[plant.corners], .008, cfg, FRAME_DT)
    from capsule_cbf.coarse_prediction import create_predictor
    from capsule_cbf.newton_runtime import CapsuleRayDetector
    from capsule_cbf.ray_basins import RayBasinController
    predictor = create_predictor(plant.model, plant.patch, plant.corner_ids, plant.offsets, 8, FRAME_DT/8,
                                 10, 16, plant.capsule, .008, cfg)
    ctrl = RayBasinController(plant.capsule, None, plant.rest[plant.corners], .008, cfg, 8,
                              CapsuleRayDetector(plant.model, plant.runtime.state, plant.capsule))
    ctrl.predictor = predictor
    return ctrl


def nominal_command(plant, cfg, leaders, corners):
    rest = plant.rest[plant.corners]
    error = corners[2:]+rest[:2]-rest[2:]-corners[:2]
    error[:, 2] = 0
    return leaders+cfg.formation_gain*error


def run(args, controller_name, scenario):
    leader_velocity, task, motion, hold = SCENARIOS[scenario]
    if controller_name == 'ray_basins' and np.any(task):
        raise ValueError('The legacy controller has no task-command input')
    cfg = Settings(**json.loads(args.config.read_text())) if args.config else Settings()
    if controller_name == 'ray_basins':
        cfg = Settings(**{**cfg.__dict__, 'controller': 'ray_basins', 'strain_enabled': False,
                          'max_speed': 0.})
    plant = build_plant(args.device, args.grid, args.areal_density, args.cloth_z,
                        iterations=args.plant_iterations, linear_iterations=args.plant_linear_iterations)
    strain = TriangleStrain(plant.model.tri_indices.numpy(), plant.model.tri_poses.numpy(),
                            cfg.weft_strain_limit, cfg.warp_strain_limit, cfg.shear_limit)
    zero = np.zeros((2, 3))
    for _ in range(args.settle_frames):
        plant.step(zero, zero)
    ctrl = make_controller(controller_name, plant, cfg)
    from capsule_cbf.cloth_model import ClothMechanics
    fine = ClothMechanics.from_newton(plant.model)
    patches = [plant.patch[plant.corner_ids == c] for c in range(4)]
    records, previous = [], np.zeros((2, 3))
    for frame in range(motion+hold):
        leaders = leader_velocity if frame < motion else zero
        q, v = plant.state()
        corners = q[plant.corners]
        nominal = nominal_command(plant, cfg, leaders, corners)+task
        begin = time.perf_counter()
        if ctrl is None:
            command, debug = nominal, dict(status='nominal')
        elif controller_name == 'sensitivity':
            command, debug = ctrl.update(ClothState(q, v), corners, leaders, FRAME_DT, nominal=nominal)
        else:
            ctrl.predictor.begin(plant.runtime.state, plant.starts)
            command, debug = ctrl.update(ctrl.predictor.current_cloth_state(), corners, leaders,
                                         FRAME_DT, ctrl.predictor)
        control_ms = 1000*(time.perf_counter()-begin)
        begin = time.perf_counter()
        plant.step(command, leaders)
        wp.synchronize_device(args.device)
        physics_ms = 1000*(time.perf_counter()-begin)
        q_next = plant.state()[0]
        d = plant.capsule.distances(q_next, .008)[0]
        m = strain.measures(q_next)
        records.append(dict(
            frame=frame, status=debug.get('status'), control_ms=control_ms, physics_ms=physics_ms,
            min_distance=float(d.min()), min_barrier=float(d.min()-cfg.margin),
            tension=[float(np.linalg.norm(fine.resultant(q_next, p))) for p in patches],
            max_tensile_strain=float(m[:, :2].max()), max_shear_cos=float(abs(m[:, 2]).max()),
            command=np.asarray(command).tolist(), nominal=nominal.tolist(),
            command_accel=float(np.abs(np.asarray(command)-previous).max()/FRAME_DT),
            correction=float(np.linalg.norm(np.asarray(command)-nominal)),
            **{k: debug[k] for k in ('candidate_count', 'total_rows', 'slack_soft', 'slack_hard',
                                     'linearize_ms', 'predict_ms', 'qp_ms', 'rows_ms', 'measure_ms',
                                     'rollout_count', 'basin_count', 'bias_norm', 'strain_ks',
                                     'worst_row_residual')
               if k in debug}))
        previous = np.asarray(command)
    return summarize(records, cfg), records


def summarize(records, cfg):
    barrier = np.array([r['min_barrier'] for r in records])
    tension = np.array([max(r['tension']) for r in records])
    control = np.array([r['control_ms'] for r in records])
    statuses = {}
    for r in records:
        statuses[r['status']] = statuses.get(r['status'], 0)+1
    return dict(frames=len(records), min_barrier=float(barrier.min()), min_distance=float(
                    min(r['min_distance'] for r in records)),
                frames_barrier_violated=int((barrier < 0).sum()),
                frames_surface_penetrated=int(sum(r['min_distance'] < 0 for r in records)),
                max_tension=float(tension.max()), tension_limit=cfg.tension_limit,
                frames_tension_violated=int((tension > cfg.tension_limit).sum()) if cfg.tension_limit else None,
                max_tensile_strain=float(max(r['max_tensile_strain'] for r in records)),
                max_command_accel=float(max(r['command_accel'] for r in records[1:])),
                mean_correction=float(np.mean([r['correction'] for r in records])),
                control_ms_p50=float(np.percentile(control, 50)), control_ms_p95=float(np.percentile(control, 95)),
                control_ms_max=float(control.max()),
                physics_ms_p50=float(np.percentile([r['physics_ms'] for r in records], 50)),
                statuses=statuses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--grid', type=int, default=56)
    p.add_argument('--areal-density', type=float, default=.2)
    p.add_argument('--cloth-z', type=float, default=.85)
    p.add_argument('--settle-frames', type=int, default=90)
    p.add_argument('--plant-iterations', type=int, default=10)
    p.add_argument('--plant-linear-iterations', type=int, default=16)
    p.add_argument('--config', type=Path, help='Settings JSON; defaults to Settings()')
    p.add_argument('--controllers', default='nominal,sensitivity')
    p.add_argument('--scenarios', default=','.join(SCENARIOS))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    wp.init()
    args.output.mkdir(parents=True, exist_ok=True)
    table = {}
    with wp.ScopedDevice(args.device):
        for scenario in args.scenarios.split(','):
            for name in args.controllers.split(','):
                summary, records = run(args, name, scenario)
                table[f'{scenario}/{name}'] = summary
                (args.output/f'{scenario}_{name}.json').write_text(json.dumps(
                    dict(summary=summary, records=records, args={k: str(v) for k, v in vars(args).items()}),
                    indent=1))
                print(f'{scenario:15s} {name:12s} ' + ' '.join(
                    f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in summary.items()),
                    flush=True)
    (args.output/'summary.json').write_text(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()

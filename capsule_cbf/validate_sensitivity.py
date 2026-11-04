"""Compare the reduced linear horizon model with the nonlinear Newton plant.

Reference sensitivities are central differences of full 56x56 Newton rollouts
(8 substeps, 10 PD / 16 PCG iterations) over the whole horizon. Run with a
Newton 1.2.x environment; see SENSITIVITY_CBF_REPORT.md.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from capsule_cbf.cloth_model import ClothMechanics
from capsule_cbf.reduced_cloth import ReducedCloth
from capsule_cbf.scenario import FRAME_DT, build_plant


def material_weights(points):
    """Rest-XY bilinear corner weights (FL, FR, LL, LR): the pre-redesign rigid influence model."""
    xy = np.asarray(points)[:, :2]
    x, y = ((xy-xy.min(axis=0))/np.ptp(xy, axis=0)).T
    return np.column_stack(((1-x)*(1-y), x*(1-y), (1-x)*y, x*y))


def rollout(plant, followers, leaders, frames):
    snapshot = plant.save()
    out = []
    for _ in range(frames):
        plant.step(followers, leaders)
        out.append(plant.state()[0])
    plant.restore(snapshot)
    return np.stack(out)


def cosine(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.sum(a*b, axis=-1)/np.maximum(na*nb, 1e-30)


def tensions(fine, patches, trajectory):
    return np.array([[np.linalg.norm(fine.resultant(q, p)) for p in patches] for q in trajectory])


def evaluate(plant, label, followers, leaders, frames, step, variants):
    q0, v0 = plant.state()
    fine = ClothMechanics.from_newton(plant.model)
    patches = [plant.patch[plant.corner_ids == c] for c in range(4)]
    started = time.perf_counter()
    nominal = rollout(plant, followers, leaders, frames)
    columns, tension_columns = [], []
    for d in np.eye(6):
        plus = rollout(plant, followers+step*d.reshape(2, 3), leaders, frames)
        minus = rollout(plant, followers-step*d.reshape(2, 3), leaders, frames)
        columns.append((plus-minus)/(2*step))
        tension_columns.append((tensions(fine, patches, plus)-tensions(fine, patches, minus))/(2*step))
    reference = np.stack(columns, axis=-1)            # (N, fine, 3, 6)
    tension_reference = np.stack(tension_columns, axis=-1)  # (N, 4, 6)
    force0 = np.array([fine.resultant(q0, p) for p in patches])
    plant_ms = 1000*(time.perf_counter()-started)
    free = np.setdiff1d(np.arange(len(q0)), plant.patch)
    d, normals, _ = plant.capsule.distances(q0, plant.particle_radius)
    near = free[np.argsort(d[free])[:64]]             # points of interest
    times = FRAME_DT*np.arange(1, frames+1)
    weights = material_weights(plant.rest)
    # Legacy heuristic: rigid bilinear influence, dp_i/du_c = t w_ic I.
    heuristic = np.zeros((frames, len(near), 3, 6))
    for c in range(2):
        heuristic[..., 3*c:3*c+3] = (times[:, None, None, None]*weights[near, c][None, :, None, None]
                                    *np.eye(3)[None, None])
    report = dict(state=label, min_clearance=float(d.min()), plant_reference_ms=plant_ms, models={})
    report['models']['bilinear_heuristic'] = summarize(reference[:, near], heuristic, normals[near])
    corner_velocity = np.vstack((followers, leaders))
    for name, kwargs in variants.items():
        red = ReducedCloth(plant.model, plant.patch, plant.corner_ids, plant.offsets, FRAME_DT, **kwargs)
        qc0, vc0 = red.restrict(q0), red.restrict(v0)
        started = time.perf_counter()
        red.horizon.linearize(qc0)
        linearize_ms = 1000*(time.perf_counter()-started)
        outputs = frames//red.stride
        out = red.horizon.predict(qc0, vc0, plant.starts, corner_velocity, outputs)
        model_ms = 1000*(time.perf_counter()-started)
        at = red.stride*np.arange(1, outputs+1)-1      # plant frame index of each output
        lifted = np.stack([red.lift_points(np.arange(len(q0)), q0, qc0, out['q'][k]) for k in range(outputs)])
        sens = np.stack([red.lift_sensitivity(near, out['sq'][k][..., :6]) for k in range(outputs)])
        error = np.linalg.norm(lifted[:, free]-nominal[at][:, free], axis=-1)
        entry = summarize(reference[at][:, near], sens, normals[near])
        # Gripper tension: input-driven model dT/du = F0_hat . J_c S (as in the controller).
        sq = out['sq'].reshape(outputs, -1, 12)[..., :6]
        model_t = np.stack([np.einsum('x,xm,nmc->nc', force0[c]/np.linalg.norm(force0[c]),
                                      red.coarse.resultant_jacobian(qc0, red.pin_nodes[red.pin_corners == c])
                                      .reshape(3, -1), sq) for c in range(4)], axis=1)  # (N, 4, 6)
        ref_t = tension_reference[at]
        entry.update(tension_gradient_cosine_median=np.median(cosine(model_t, ref_t), axis=1).tolist(),
                     tension_gradient_norm_ratio_median=np.median(
                         np.linalg.norm(model_t, axis=-1)/np.maximum(np.linalg.norm(ref_t, axis=-1), 1e-12),
                         axis=1).tolist())
        entry.update(model_ms=model_ms, linearize_ms=linearize_ms, output_frames=(at+1).tolist(),
                     drift_max_error_m=[float(e.max()) for e in error],
                     drift_rms_error_m=[float(np.sqrt(np.mean(e**2))) for e in error],
                     plant_motion_max_m=[float(np.linalg.norm(nominal[k, free]-q0[free], axis=-1).max())
                                         for k in at], **red.info)
        report['models'][name] = entry
    return report


def summarize(reference, model, normals):
    """Errors of position and clearance sensitivities, per horizon frame."""
    rel = (np.linalg.norm((model-reference).reshape(len(reference), -1), axis=1)
           /np.maximum(np.linalg.norm(reference.reshape(len(reference), -1), axis=1), 1e-30))
    block_cos = cosine(model.reshape(*model.shape[:2], -1), reference.reshape(*reference.shape[:2], -1))
    dh_ref = np.einsum('px,kpxc->kpc', normals, reference)
    dh_model = np.einsum('px,kpxc->kpc', normals, model)
    dh_cos = cosine(dh_model, dh_ref)
    dh_rel = np.linalg.norm(dh_model-dh_ref, axis=-1)/np.maximum(np.linalg.norm(dh_ref, axis=-1), 1e-30)
    pick = lambda a: [float(a[k]) for k in range(len(a))]
    return dict(position_sensitivity_rel_error=pick(rel),
                position_block_cosine_median=pick(np.median(block_cos, axis=1)),
                clearance_gradient_cosine_median=pick(np.median(dh_cos, axis=1)),
                clearance_gradient_cosine_p10=pick(np.percentile(dh_cos, 10, axis=1)),
                clearance_gradient_rel_error_median=pick(np.median(dh_rel, axis=1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--grid', type=int, default=56)
    p.add_argument('--areal-density', type=float, default=.2)
    p.add_argument('--cloth-z', type=float, default=.85)
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--settle-frames', type=int, default=120)
    p.add_argument('--fd-step', type=float, default=.01)
    p.add_argument('--plant-iterations', type=int, default=10)
    p.add_argument('--plant-linear-iterations', type=int, default=16)
    p.add_argument('--all-variants', action='store_true', help='Also run the model-resolution study')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    wp.init()
    with wp.ScopedDevice(args.device):
        plant = build_plant(args.device, args.grid, args.areal_density, args.cloth_z,
                            iterations=args.plant_iterations,
                            linear_iterations=args.plant_linear_iterations)
        zero = np.zeros((2, 3))
        for _ in range(args.settle_frames):
            plant.step(zero, zero)
        variants = {'coarse14_affine_stride2': dict(cells=14, stride=2)}
        if args.all_variants:
            variants.update({'coarse14_affine_stride1': dict(cells=14),
                             'coarse14_affine_stride2_sub2': dict(cells=14, stride=2, substeps=2),
                             'coarse28_affine_stride1': dict(cells=28),
                             'fine_affine_stride1': dict(cells=0),
                             'coarse14_trajectory_stride1': dict(cells=14, jacobian='trajectory')})
        reports = [evaluate(plant, 'settled', zero, zero, args.frames, args.fd_step, variants)]
        descend = np.tile([0., 0., -.1], (2, 1))
        for _ in range(20):
            plant.step(descend, descend)
        reports.append(evaluate(plant, 'descending_0.1mps', descend, descend, args.frames,
                                args.fd_step, variants))
    result = dict(grid=args.grid, areal_density=args.areal_density, cloth_z=args.cloth_z,
                  plant_iterations=args.plant_iterations,
                  plant_linear_iterations=args.plant_linear_iterations,
                  frames=args.frames, fd_step=args.fd_step, reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for r in reports:
        print('==', r['state'], 'min clearance %.3f' % r['min_clearance'])
        for name, m in r['models'].items():
            frames = m.get('output_frames', list(range(1, len(m['clearance_gradient_cosine_median'])+1)))
            k = [frames.index(f) for f in (frames[0], 6, 10, frames[-1]) if f in frames]
            print(f"{name:30s} frames {[frames[i] for i in k]}"
                  f" dh/du cos med {[round(m['clearance_gradient_cosine_median'][i], 3) for i in k]}"
                  f" relerr {[round(m['clearance_gradient_rel_error_median'][i], 3) for i in k]}"
                  + (f" drift max mm {[round(1000*m['drift_max_error_m'][i], 2) for i in k]}"
                     f" ms {m['model_ms']:.1f}" if 'model_ms' in m else ''))
            if 'tension_gradient_cosine_median' in m:
                print(f"{'':30s} dT/du cos med {[round(m['tension_gradient_cosine_median'][i], 3) for i in k]}"
                      f" |model|/|plant| {[round(m['tension_gradient_norm_ratio_median'][i], 2) for i in k]}")


if __name__ == '__main__':
    main()

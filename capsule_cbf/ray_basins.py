"""Ray hit grouping and one discrete-time obstacle barrier per basin.

Ray traversal is GPU-side; grouping of at most atlas-size hits is CPU-side.
Representatives retain triangle barycentric coordinates throughout a rollout.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from .core import solve_filter
from .predictive import six_direction_sensitivity, dynamics_sensitivity
from .dynamics_sensitivity import SensitivityUnavailable
from .cloth_state import unpack_cloth_state


def basin_representatives(points, distances, k, radius):
    """Connected components of radius-limited, undirected k-nearest edges.

    Choose the minimum-clearance sample, never an averaged position. Radius
    prevents isolated surfaces from being merged just because k neighbors exist.
    """
    n = len(points)
    if not n:
        return np.empty(0, dtype=int)
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ds, js = cKDTree(points).query(points, k=min(n, k+1))
    for i, (dd, jj) in enumerate(zip(np.asarray(ds).reshape(n, -1),
                                    np.asarray(js).reshape(n, -1))):
        for d, j in zip(dd, jj):
            if i != j and d <= radius:
                parent[root(int(j))] = root(i)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return np.array([min(ids, key=lambda i: (distances[i], i))
                     for ids in groups.values()], dtype=int)


class RayBasinController:
    def __init__(self, capsule, strain, rest_corners, particle_radius, settings,
                 substeps, detector):
        if settings.strain_enabled:
            raise ValueError('Ray basin mode currently requires strain_enabled=false; '
                             'use ray_basins_enabled=false for full strain control')
        self.capsule, self.strain, self.settings = capsule, strain, settings
        self.rest_corners = np.asarray(rest_corners).copy()
        self.particle_radius, self.substeps = particle_radius, substeps
        self.detector = detector

    def update(self, points, corners, leaders, dt, rollout):
        started = time.perf_counter()
        points, state_debug = unpack_cloth_state(points, self.capsule)
        cfg = self.settings
        points, corners, leaders = [np.asarray(x, float) for x in (points, corners, leaders)]
        if (corners.shape != (4, 3) or leaders.shape != (2, 3) or dt <= 0
                or not np.isfinite(dt) or not all(np.isfinite(x).all() for x in (points, corners, leaders))):
            raise ValueError('Invalid controller input')
        hits = self.detector.detect()
        distances, _, surfaces = self.capsule.distances(hits['points'], self.particle_radius)
        active = np.flatnonzero(distances <= cfg.margin)
        detected = hits['points'][active]
        chosen = basin_representatives(detected, distances[active], cfg.basin_knn,
                                       cfg.basin_neighbor_radius)
        support = active[chosen]
        current = distances[support]-cfg.margin
        # Leaders are indices 2:4, followers 0:2; preserve only planar formation.
        error = corners[2:]+self.rest_corners[:2]-self.rest_corners[2:]-corners[:2]
        if not cfg.vertical_formation_recovery:
            error[:, 2] = 0
        nominal = leaders+cfg.formation_gain*error
        if cfg.max_speed > 0:
            nominal *= np.minimum(1., cfg.max_speed/np.maximum(np.linalg.norm(nominal, axis=1), 1e-12))[:, None]
        rollout.select_hits(hits['faces'][support], hits['weights'][support])
        decay = np.exp(-cfg.alpha*dt*np.arange(1, self.substeps+1)/self.substeps)
        target = decay[:, None]*current+(1-decay[:, None])*cfg.clearance_tightening
        calls, rollout_ms, optimization_ms = 0, 0., 0.
        sensitivity_ms, sensitivity_modes, sensitivity_fallbacks = 0., [], []
        last_prediction = None
        def residual(u, _):
            nonlocal calls, rollout_ms, last_prediction
            begin = time.perf_counter()
            predicted = np.asarray(rollout(u, leaders), float)
            calls += 1
            rollout_ms += 1000*(time.perf_counter()-begin)
            if predicted.shape != target.shape or not np.isfinite(predicted).all():
                raise ValueError('Invalid selected-hit prediction')
            last_prediction = predicted
            # ONE row per basin, covering the worst substep of that material hit.
            return np.min(predicted-target, axis=0)
        rows, rhs = np.empty((0, 6)), np.empty(0)
        center = nominal.copy()
        best = nominal.copy()
        base = residual(center, leaders) if len(support) else np.empty(0)
        violation = float(np.maximum(-base, 0).max(initial=0))
        initial_violation = violation
        best_prediction = last_prediction
        status = 'feasible' if violation <= 1e-6 else 'prediction_not_safe'
        linear_status = 'not_needed'
        for _ in range(cfg.prediction_iterations):
            if status == 'feasible':
                break
            tangent_started = time.perf_counter()
            jac = None
            linear_base = base
            if cfg.sensitivity_method != 'finite_difference':
                try:
                    jac = dynamics_sensitivity(rollout, center, leaders, target.shape)
                    # Keep every substep: no derivative of a nonsmooth min and
                    # no loss of a competing substep near a tie.
                    linear_base = (last_prediction-target).ravel()
                    jac = jac.reshape(-1, 6)
                    sensitivity_modes.append(cfg.sensitivity_method)
                except SensitivityUnavailable as error:
                    sensitivity_fallbacks.append(str(error))
            if jac is None:
                jac = six_direction_sensitivity(residual, center, leaders, cfg.sensitivity_step, base)
                sensitivity_modes.append('finite_difference')
            sensitivity_ms += 1000*(time.perf_counter()-tangent_started)
            rows = jac/.01
            rhs = (-linear_base+jac@center.ravel())/.01
            trust = cfg.prediction_trust_displacement/dt
            bounds = np.column_stack((center.ravel()-trust, center.ravel()+trust))
            begin = time.perf_counter()
            trial, linear_status, _ = solve_filter(nominal, rows, rhs, cfg.max_speed, bounds)
            optimization_ms += 1000*(time.perf_counter()-begin)
            checked = residual(trial, leaders)
            trial_violation = float(np.maximum(-checked, 0).max(initial=0))
            if trial_violation < violation:
                best, violation = trial.copy(), trial_violation
                best_prediction = last_prediction
            if trial_violation <= 1e-6:
                best, status = trial, 'feasible'
                best_prediction = last_prediction
                break
            center, base = trial, checked
        all_d = self.capsule.distances(points, self.particle_radius)[0]
        debug = dict(status=status, mode='newton_ray_basins', strain_enabled=False, **state_debug,
                     min_clearance=float(all_d.min()), h_min=float(all_d.min()-cfg.margin),
                     initially_safe=bool(np.all(all_d >= cfg.margin)),
                     selected_initially_safe=bool(np.all(current >= 0)),
                     predicted_nonnegative=(bool(np.all(best_prediction >= 0))
                                            if best_prediction is not None else None),
                     predicted_selected_min_clearance=(float(best_prediction.min()+cfg.margin)
                                                       if best_prediction is not None else None),
                     total_rows=len(rows), basin_rows=len(support), occupied_cells=len(active),
                     ray_hits=len(hits['points']), basin_count=len(support),
                     max_violation=violation, nominal_prediction_violation=initial_violation,
                     correction=float(np.linalg.norm(best-nominal)), linear_status=linear_status,
                     rollout_count=calls, rollout_ms=rollout_ms, optimization_ms=optimization_ms,
                     sensitivity_ms=sensitivity_ms, sensitivity_modes=sensitivity_modes,
                     sensitivity_fallbacks=sensitivity_fallbacks,
                     controller_ms=1000*(time.perf_counter()-started),
                     detected_points=detected, basin_points=hits['points'][support],
                     detected_indices=[], basin_indices=[], surfaces=surfaces[active],
                     rows=rows, rhs=rhs,
                     follower_velocity=best.tolist(), nominal_velocity=nominal.tolist(),
                     safety_scope='Selected ray basin material hits only; full mesh safety is not certified')
        debug.update(getattr(rollout, 'resolution_info', {}))
        if debug.get('prediction_model') == 'coarse':
            debug['safety_scope'] = ('Selected fine-mesh ray hits predicted with coarse dynamics; '
                                     'coarse feasibility does not certify full-mesh safety')
        return best, debug

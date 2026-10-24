"""Six-input sequential linearization with full nonlinear rollout validation.

The rollout callback is replaceable for CPU tests. No Isaac/Warp imports here.
"""
import time
import numpy as np
from .core import solve_filter
from .dynamics_sensitivity import SensitivityUnavailable
from .cloth_state import unpack_cloth_state


def dynamics_sensitivity(rollout, reference, leaders, shape):
    """Validate the optional predictor interface; programming errors propagate."""
    if not hasattr(rollout, 'sensitivity'):
        raise SensitivityUnavailable('predictor has no sensitivity interface')
    jac = np.asarray(rollout.sensitivity(reference, leaders), dtype=float)
    if jac.shape != (*shape, 6) or not np.isfinite(jac).all():
        raise SensitivityUnavailable('invalid dynamics observation Jacobian')
    return jac


def six_direction_sensitivity(rollout, reference, leaders, step, baseline=None):
    """Forward differences of every observed substep barrier, six DOFs only."""
    reference = np.asarray(reference, dtype=float).reshape(2, 3)
    base = rollout(reference, leaders) if baseline is None else baseline
    columns = []
    for direction in np.eye(6):
        perturbed = reference+step*direction.reshape(2, 3)
        columns.append((rollout(perturbed, leaders)-base)/step)
    return np.stack(columns, axis=-1)


class PredictiveController:
    def __init__(self, capsule, strain, rest_corners, particle_radius, settings,
                 substeps):
        self.capsule, self.strain, self.settings = capsule, strain, settings
        self.rest_corners = np.asarray(rest_corners, dtype=float).copy()
        self.particle_radius, self.substeps = particle_radius, substeps
        self.cache = None
        self.frame = 0

    def barriers(self, points):
        d, _, _ = self.capsule.distances(points, self.particle_radius)
        return (np.concatenate((d-self.settings.margin, self.strain.barriers(points).ravel()))
                if self.settings.strain_enabled else d-self.settings.margin)

    def update(self, points, corners, leaders, dt, rollout):
        started = time.perf_counter()
        points, state_debug = unpack_cloth_state(points, self.capsule)
        cfg = self.settings
        points, corners, leaders = [np.asarray(x, dtype=float) for x in (points, corners, leaders)]
        if (corners.shape != (4, 3) or leaders.shape != (2, 3) or dt <= 0
                or not np.isfinite(dt) or not all(np.isfinite(x).all() for x in (points, corners, leaders))):
            raise ValueError('Invalid predictive controller input')
        n = len(points)
        current = self.barriers(points)
        decay = np.exp(-cfg.alpha*dt*np.arange(1, self.substeps+1)/self.substeps)
        tighten = np.r_[np.full(n, cfg.clearance_tightening),
                        np.full(len(current)-n, cfg.strain_tightening)]
        # Tightening scales with elapsed time; no discontinuity at t=0.
        target = decay[:, None]*current+(1-decay[:, None])*tighten
        activation = np.r_[np.full(n, cfg.activation),
                           np.full(len(current)-n, cfg.strain_activation)]
        offset = self.rest_corners[:2]-self.rest_corners[2:]
        error = corners[2:]+offset-corners[:2]
        if not cfg.vertical_formation_recovery:
            error[:, 2] = 0
        nominal = leaders+cfg.formation_gain*error
        center = nominal.copy()
        if cfg.max_speed > 0:
            center *= np.minimum(1., cfg.max_speed/np.maximum(np.linalg.norm(center, axis=1), 1e-12))[:, None]
        calls = 0
        rollout_ms = 0.
        optimization_ms = 0.
        sensitivity_ms, sensitivity_modes, sensitivity_fallbacks = 0., [], []
        def predict(u):
            nonlocal calls, rollout_ms
            calls += 1
            begin = time.perf_counter()
            result = np.asarray(rollout(u, leaders), dtype=float)
            rollout_ms += 1000*(time.perf_counter()-begin)
            if result.shape != target.shape or not np.isfinite(result).all():
                raise ValueError('Nonfinite or incorrectly shaped Newton prediction')
            return result
        base = predict(center)
        initial_violation = float(np.maximum(target-base, 0).max())
        best, best_prediction, best_violation = center.copy(), base.copy(), initial_violation
        rows, rhs = np.empty((0, 6)), np.empty(0)
        reused, refreshed = False, False
        linear_status = 'not_needed'
        cache_ok = (cfg.sensitivity_method == 'finite_difference' and self.cache is not None
                    and self.frame-self.cache['frame'] < cfg.sensitivity_refresh_frames
                    and np.linalg.norm(points-self.cache['points'], axis=1).max() <= cfg.sensitivity_state_threshold
                    and np.linalg.norm(leaders-self.cache['leaders']) <= cfg.sensitivity_command_threshold
                    and np.linalg.norm(center-self.cache['center']) <= cfg.sensitivity_command_threshold)
        status = 'feasible' if initial_violation <= 1e-6 else 'prediction_not_safe'
        selected_mask = np.zeros_like(target, dtype=bool)
        for iteration in range(cfg.prediction_iterations):
            if status == 'feasible':
                break
            if cache_ok and iteration == 0:
                jacobian = self.cache['jacobian']
                reused = True
            else:
                tangent_started = time.perf_counter()
                jacobian = None
                if cfg.sensitivity_method != 'finite_difference':
                    try:
                        jacobian = dynamics_sensitivity(rollout, center, leaders, base.shape)
                        sensitivity_modes.append(cfg.sensitivity_method)
                    except SensitivityUnavailable as error:
                        sensitivity_fallbacks.append(str(error))
                if jacobian is None:
                    jacobian = six_direction_sensitivity(lambda u, _: predict(u), center,
                                                           leaders, cfg.sensitivity_step, base)
                    sensitivity_modes.append('finite_difference')
                sensitivity_ms += 1000*(time.perf_counter()-tangent_started)
                self.cache = dict(jacobian=jacobian, points=points.copy(), leaders=leaders.copy(),
                                  center=center.copy(), frame=self.frame)
                refreshed = True
            # Activation is proximity to a physical barrier, not proximity to
            # its discrete-time decrease bound (which is small everywhere).
            mask = ((base <= activation[None, :])
                    | (current[None, :] <= activation[None, :])
                    | (target-base > 1e-6))
            # Keep rows that were repaired in an earlier iteration; otherwise
            # minimizing against nominal can undo that repair on the next one.
            selected_mask |= mask
            mask = selected_mask
            rows = jacobian[mask]
            rhs = (target-base)[mask]+np.einsum('ij,j->i', rows, center.ravel(), optimize=False)
            # Normalize by 1 cm clearance / 1% engineering strain so tiny
            # per-frame sensitivities do not dominate optimizer tolerances.
            scales = np.broadcast_to(np.r_[np.full(n, .01),
                                           np.full(len(current)-n, .01)], target.shape)[mask]
            rows, rhs = rows/scales[:, None], rhs/scales
            redundant = (np.linalg.norm(rows, axis=1) < 1e-12) & (rhs <= 0)
            rows, rhs = rows[~redundant], rhs[~redundant]
            trust = cfg.prediction_trust_displacement/dt
            bounds = np.column_stack((center.ravel()-trust, center.ravel()+trust))
            optimization_started = time.perf_counter()
            trial, linear_status, _ = solve_filter(nominal, rows, rhs, cfg.max_speed, bounds)
            optimization_ms += 1000*(time.perf_counter()-optimization_started)
            predicted = predict(trial)
            violation = float(np.maximum(target-predicted, 0).max())
            if violation < best_violation:
                best, best_prediction, best_violation = trial.copy(), predicted.copy(), violation
            if violation <= 1e-6:
                best, best_prediction, best_violation = trial, predicted, violation
                status = 'feasible'
                break
            # A failed nonlinear check invalidates reuse on the next iteration.
            self.cache = None
            center, base = trial, predicted
        if status != 'feasible':
            self.cache = None
        self.frame += 1
        d, _, surface = self.capsule.distances(points, self.particle_radius)
        active = np.flatnonzero(current[:n] <= cfg.activation).tolist()
        debug = dict(status=status, mode='newton_predictive', **state_debug,
                     initially_safe=bool(np.all(current >= 0)),
                     predicted_nonnegative=bool(np.all(best_prediction >= 0)),
                     min_clearance=float(d.min()), h_min=float(current[:n].min()),
                     total_rows=len(rows), basin_rows=0, occupied_cells=0,
                     max_violation=best_violation, nominal_prediction_violation=initial_violation,
                     max_clearance_violation=float(np.maximum(target[:, :n]-best_prediction[:, :n], 0).max()),
                     max_strain_violation=float(np.maximum(target[:, n:]-best_prediction[:, n:], 0).max(initial=0)),
                     linear_status=linear_status, rollout_count=calls,
                     sensitivity_reused=reused, sensitivity_refreshed=refreshed,
                     sensitivity_ms=sensitivity_ms, sensitivity_modes=sensitivity_modes,
                     sensitivity_fallbacks=sensitivity_fallbacks,
                     controller_ms=1000*(time.perf_counter()-started),
                     rollout_ms=rollout_ms, optimization_ms=optimization_ms,
                     predicted_min_clearance=float(best_prediction[:, :n].min()+cfg.margin),
                     predicted_min_strain_barrier=float(best_prediction[:, n:].min()) if cfg.strain_enabled else None,
                     correction=float(np.linalg.norm(best-nominal)),
                     detected_indices=active, basin_indices=[], surfaces=surface[active],
                     rows=rows, rhs=rhs, **self.strain.summary(points))
        return best, debug

"""Sensitivity-based multi-step discrete CBF filter for the two follower grippers.

Each update (see SENSITIVITY_CBF_REPORT.md for derivations and sources):
 1. measure the fine cloth (q, v), restrict to the coarse model and update a
    force-bias disturbance observer (offset-free prediction);
 2. linearize once and predict N outputs with 12 corner-velocity sensitivities;
 3. outputs y: capsule clearance at rest-mesh local minima, gripper tension and
    KS gripper-band strain, each anchored at its fine-mesh value;
 4. rows  y_k(u) >= exp(-alpha t_k) y_0 + eps_k,  plus h_N + tau dh_N/dt >= eps_N,
    eps_k = online conformal quantile of model residuals + leader-acceleration bound;
 5. hierarchical QP: stretch rows hard, clearance rows soft, follower rate box.
No Isaac, ROS or Warp imports.
"""
import math
import time
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix, identity
from .cloth_state import unpack_cloth_state
from .qp import safety_qp
from .strain import TriangleStrain

CLEARANCE, TENSION, STRAIN = 'clearance', 'tension', 'strain'


def rest_rings(triangles, count, rings):
    """Padded neighbor table of all nodes within `rings` mesh edges (self included)."""
    t = np.asarray(triangles, dtype=int)
    i = np.concatenate((t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]))
    j = np.concatenate((t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]))
    a = (csr_matrix((np.ones(len(i)), (i, j)), shape=(count, count))+identity(count)).tocsr()
    reach = a
    for _ in range(rings-1):
        reach = reach@a
    reach = reach.tocsr()
    counts = np.diff(reach.indptr)
    rows = np.repeat(np.arange(count), counts)
    table = np.repeat(np.arange(count)[:, None], counts.max(), axis=1)
    table[rows, np.arange(len(rows))-reach.indptr[rows]] = reach.indices
    return table


def local_minima(values, table):
    """Strict minima under the order (value, index): one node per plateau."""
    index = np.arange(len(values))
    neighbor = values[table]
    lower = (values[:, None] < neighbor) | ((values[:, None] == neighbor) & (index[:, None] <= table))
    return np.flatnonzero(lower.all(axis=1))


def band(triangles, rest, patch, radius, pinned):
    """Unpinned triangles within `radius` (rest plane) of a gripper patch."""
    centroid = rest[triangles].mean(axis=1)[:, :2]
    gap = np.linalg.norm(centroid[:, None, :]-rest[patch][None, :, :2], axis=2).min(axis=1)
    return np.flatnonzero((gap <= radius) & ~pinned[triangles].all(axis=1))


class ConformalBound:
    """Sliding-window split-conformal upper quantile of one-sided residuals.

    eps = the ceil((n+1)q)-th smallest residual (Vovk et al. 2005; Angelopoulos
    and Bates 2023). Closed-loop data are not exchangeable, so the window
    gives adaptivity (Gibbs and Candes 2021) rather than a formal guarantee.
    """
    def __init__(self, outputs, window, quantile, fallback, minimum=20):
        self.buffers = [deque(maxlen=window) for _ in range(outputs)]
        self.quantile, self.fallback, self.minimum = quantile, fallback, minimum

    def add(self, k, residuals):
        self.buffers[k].extend(np.atleast_1d(residuals).tolist())

    def bound(self, k):
        data = self.buffers[k]
        if len(data) < self.minimum:
            return self.fallback
        rank = math.ceil((len(data)+1)*self.quantile)
        value = max(data) if rank > len(data) else np.partition(np.asarray(data), rank-1)[rank-1]
        return max(float(value), 0.)

    def bounds(self):
        return np.array([self.bound(k) for k in range(len(self.buffers))])


class SensitivityCbfController:
    def __init__(self, capsule, reduced, rest_corners, particle_radius, settings, frame_dt):
        cfg = self.settings = settings
        self.capsule, self.red = capsule, reduced
        self.rest_corners = np.asarray(rest_corners, dtype=float).copy()
        self.particle_radius, self.frame_dt = particle_radius, frame_dt
        self.stride = reduced.stride
        if cfg.horizon_stride != self.stride:
            raise ValueError('Reduced model stride must equal horizon_stride')
        self.outputs = cfg.horizon_frames//self.stride
        self.times = frame_dt*self.stride*np.arange(1, self.outputs+1)
        fine, coarse = reduced.fine, reduced.coarse
        self.neighbors = rest_rings(fine.triangles, fine.count, cfg.candidate_rings)
        self.fine_patches = [reduced.fine_patch[reduced.fine_corner_ids == c] for c in range(4)]
        self.coarse_patches = [reduced.pin_nodes[reduced.pin_corners == c] for c in range(4)]
        limits = (cfg.weft_strain_limit, cfg.warp_strain_limit, cfg.shear_limit)
        self.fine_strain = TriangleStrain(fine.triangles, fine.inv_dm, *limits)
        self.coarse_strain = TriangleStrain(coarse.triangles, coarse.inv_dm, *limits)
        coarse_rest = reduced.rest[reduced.source]
        self.fine_bands = [band(fine.triangles, reduced.rest, self.fine_patches[c], cfg.strain_band_radius,
                                fine.pinned) for c in range(4)]
        self.coarse_bands = [band(coarse.triangles, coarse_rest, self.coarse_patches[c],
                                  cfg.strain_band_radius, coarse.pinned) for c in range(4)]
        if cfg.strain_enabled and min(len(b) for b in self.coarse_bands) == 0:
            raise ValueError('strain_band_radius contains no coarse triangle at some gripper')
        self.conformal = {CLEARANCE: ConformalBound(self.outputs, cfg.conformal_window, cfg.conformal_quantile,
                                                    cfg.initial_clearance_margin),
                          TENSION: ConformalBound(self.outputs, cfg.conformal_window, cfg.conformal_quantile,
                                                  cfg.initial_tension_margin),
                          STRAIN: ConformalBound(self.outputs, cfg.conformal_window, cfg.conformal_quantile,
                                                 cfg.initial_strain_margin)}
        self.u_prev = np.zeros((2, 3))
        self.bias = None
        self.frame = 0
        self.history = {}  # frame -> applied 4x3 corner velocities
        self.pending = {}  # arrival frame -> predictions to score
        self.observed = deque()

    # ---------------------------------------------------------------- outputs
    def _measure(self, q):
        cfg = self.settings
        d, normals, surfaces = self.capsule.distances(q, self.particle_radius)
        tension = np.array([np.linalg.norm(self.red.fine.resultant(q, p)) for p in self.fine_patches])
        strain = np.array([self.fine_strain.aggregate(q, b, cfg.ks_kappa)[0] if len(b) else 0.
                           for b in self.fine_bands])
        return d, surfaces, tension, strain

    def _clearance_rows(self, q, h_now, qc, out):
        cfg, red = self.settings, self.red
        n_out = self.outputs
        dq = out['q']-qc
        h_min = h_now.copy()
        # Only nodes that could reach the activation band need lifting.
        reach = np.linalg.norm(dq, axis=2).max(initial=0.)
        near = np.flatnonzero(h_now <= cfg.prediction_activation+reach)
        lift = red.lift[near]
        for k in (n_out//2, n_out-1):
            lifted = q[near]+lift@dq[k]
            h_min[near] = np.minimum(h_min[near],
                                     self.capsule.distances(lifted, self.particle_radius)[0]-cfg.margin)
        minima = local_minima(h_min, self.neighbors)
        minima = minima[h_min[minima] <= cfg.prediction_activation]
        idx = minima[np.argsort(h_min[minima], kind='stable')][:cfg.max_candidates]
        if not len(idx):
            return idx, np.empty((0, n_out)), np.empty((0, n_out, 12)), np.empty(0), np.empty((0, 12)), h_min
        w = red.lift[idx]
        points = q[idx][None]+np.stack([w@dq[k] for k in range(n_out)])          # (N, K, 3)
        d, normals, _ = self.capsule.distances(points.reshape(-1, 3), self.particle_radius)
        normals = normals.reshape(n_out, len(idx), 3)
        y = (d.reshape(n_out, len(idx))-cfg.margin).T                              # (K, N)
        sens = np.stack([(w@out['sq'][k].reshape(len(qc), -1)).reshape(len(idx), 3, 12)
                         for k in range(n_out)])                                   # (N, K, 3, 12)
        g = np.einsum('nkx,nkxc->knc', normals, sens)
        vel = w@out['v'][-1]
        sv = (w@out['sv'][-1].reshape(len(qc), -1)).reshape(len(idx), 3, 12)
        tau = cfg.terminal_lookahead
        y_term = y[:, -1]+tau*np.einsum('kx,kx->k', normals[-1], vel)
        g_term = g[:, -1]+tau*np.einsum('kx,kxc->kc', normals[-1], sv)
        return idx, y, g, y_term, g_term, h_min

    # Stretch outputs are spatial derivatives of position: on 5 cm coarse
    # elements a 1 mm drift error is ~2 % strain, the size of the limit. They
    # are therefore predicted from the measured fine value plus the
    # input-driven response only, y_k = y_now + (dy/dq) S_k [u; w]; free cloth
    # drift (settling, swing) is left to the conformal residual bound. A
    # velocity-form variant (measured rate + S_k on the command change) was
    # tested and destabilized the loop (report, section 6.2).
    def _tension_rows(self, qc, out, velocity):
        cfg, red = self.settings, self.red
        sens = out['sq'].reshape(self.outputs, -1, 12)
        values, grads = [], []
        for c in range(4):
            jac = red.coarse.resultant_jacobian(qc, self.coarse_patches[c]).reshape(3, -1)
            force0 = red.fine.resultant(self._q, self.fine_patches[c])
            dforce = np.einsum('xm,nmc->nxc', jac, sens)
            force = force0[None]+dforce@velocity
            norm = np.maximum(np.linalg.norm(force, axis=1), 1e-12)
            values.append(cfg.tension_limit-norm)
            grads.append(-np.einsum('nx,nxc->nc', force/norm[:, None], dforce))
        return np.array(values), np.array(grads)

    def _strain_rows(self, qc, out, strain_now, velocity):
        cfg = self.settings
        values, grads = [], []
        for c in range(4):
            grad = self.coarse_strain.aggregate(qc, self.coarse_bands[c], cfg.ks_kappa)[1]
            dks = np.einsum('mx,nmxc->nc', grad, out['sq'])
            values.append(1.-(strain_now[c]+dks@velocity))
            grads.append(-dks)
        return np.array(values), np.array(grads)

    # ------------------------------------------------------------- bookkeeping
    def _mean_velocity(self, start, stop):
        rows = [self.history[f] for f in range(start, stop) if f in self.history]
        return np.mean(rows, axis=0).ravel() if rows else None

    def _score(self, h_now, tension_now, strain_now, qc):
        """Residuals of predictions arriving now, corrected for command changes."""
        for kind, key, k, made, predicted, grad, assumed in self.pending.pop(self.frame, []):
            actual = self._mean_velocity(made, self.frame)
            if actual is not None:
                predicted = predicted+grad@(actual-assumed)
            measured = (h_now[key] if kind == CLEARANCE else
                        self.settings.tension_limit-tension_now[key] if kind == TENSION else
                        1.-strain_now[key])
            self.conformal[kind].add(k, predicted-measured)
        while self.observed and self.observed[0][0] <= self.frame:
            arrival, made, predicted, sens, assumed = self.observed.popleft()
            if arrival < self.frame:
                continue
            actual = self._mean_velocity(made, self.frame)
            if actual is not None:
                predicted = predicted+(sens@(actual-assumed)).reshape(predicted.shape)
            hz = self.red.horizon
            correction = self.settings.observer_gain*hz.one_step_response(qc-predicted)
            self.bias = correction if self.bias is None else self.bias+correction
            # A force bias above twice the weight indicates an unmodelled event
            # (contact, grasp change), not a disturbance worth integrating.
            cap = 2*9.81*hz.mass_free
            self.bias = np.clip(self.bias, -cap, cap)

    # ------------------------------------------------------------------ update
    def update(self, cloth_state, corners, leaders, dt, nominal=None):
        """nominal: optional task command (2, 3); defaults to leader tracking."""
        started = time.perf_counter()
        cfg, red, hz = self.settings, self.red, self.red.horizon
        if hasattr(cloth_state, 'velocities'):
            q = np.asarray(cloth_state.positions, dtype=float)
            v = np.asarray(cloth_state.velocities, dtype=float)
            state_debug = dict(state_velocity_available=True)
        else:  # position-only callers: zero velocity, flagged in diagnostics
            q, state_debug = unpack_cloth_state(cloth_state, self.capsule)
            q = np.asarray(q, dtype=float)
            v = np.zeros_like(q)
        corners, leaders = np.asarray(corners, dtype=float), np.asarray(leaders, dtype=float)
        if (q.shape != (red.fine.count, 3) or corners.shape != (4, 3) or leaders.shape != (2, 3)
                or not all(np.isfinite(x).all() for x in (q, v, corners, leaders))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Invalid controller input')
        self._q = q
        timing = {}
        d, surfaces, tension_now, strain_now = self._measure(q)
        h_now = d-cfg.margin
        qc, vc = red.restrict(q), red.restrict(v)
        self._score(h_now, tension_now, strain_now, qc)
        timing['measure_ms'] = 1000*(time.perf_counter()-started)

        if nominal is None:
            error = corners[2:]+self.rest_corners[:2]-self.rest_corners[2:]-corners[:2]
            if not cfg.vertical_formation_recovery:
                error[:, 2] = 0
            nominal = leaders+cfg.formation_gain*error
        nominal = np.array(nominal, dtype=float).reshape(2, 3)
        speed = cfg.max_speed if cfg.max_speed > 0 else np.inf
        nominal *= np.minimum(1., speed/np.maximum(np.linalg.norm(nominal, axis=1), 1e-12))[:, None]
        u_ref = np.clip(self.u_prev, -speed, speed)
        velocity = np.vstack((u_ref, leaders))

        begin = time.perf_counter()
        hz.linearize(qc)
        timing['linearize_ms'] = 1000*(time.perf_counter()-begin)
        begin = time.perf_counter()
        bias = None if self.bias is None or cfg.observer_gain == 0 else self.bias
        out = hz.predict(qc, vc, corners, velocity, self.outputs, bias)
        timing['predict_ms'] = 1000*(time.perf_counter()-begin)

        begin = time.perf_counter()
        idx, y_clear, g_clear, y_term, g_term, h_min = self._clearance_rows(q, h_now, qc, out)
        blocks = []  # (kind, keys, y (rows, N), grad (rows, N, 12), y0, alpha, group)
        if len(idx):
            blocks.append((CLEARANCE, idx, y_clear, g_clear, h_now[idx], cfg.alpha, 0))
        if cfg.tension_limit > 0:
            y_t, g_t = self._tension_rows(qc, out, velocity.ravel())
            blocks.append((TENSION, np.arange(4), y_t, g_t, cfg.tension_limit-tension_now, cfg.stretch_alpha, 1))
        if cfg.strain_enabled:
            y_s, g_s = self._strain_rows(qc, out, strain_now, velocity.ravel())
            blocks.append((STRAIN, np.arange(4), y_s, g_s, 1.-strain_now, cfg.stretch_alpha, 1))
        decay_t = self.times
        leader_eps = .5*cfg.leader_max_accel*decay_t
        rows, rhs, groups, records, kinds = [], [], [], [], []
        lower = np.full(6, -speed)
        upper = np.full(6, speed)
        if cfg.follower_max_accel > 0:
            step = cfg.follower_max_accel*dt
            lower = np.maximum(lower, u_ref.ravel()-step)
            upper = np.minimum(upper, u_ref.ravel()+step)
        # Dimensionless rows, so one group slack compares like with like:
        # 1 unit = 1 cm clearance, 5 % of the tension limit, 5 % of the strain limit.
        scales = {CLEARANCE: .01, TENSION: .05*max(cfg.tension_limit, 1e-9), STRAIN: .05}
        for kind, keys, y, g, y0, alpha, group in blocks:
            eps = np.broadcast_to(self.conformal[kind].bounds()[None, :], y.shape)
            if kind == CLEARANCE:
                # Worst-case leader acceleration. Not applied to stretch: on a
                # stiff membrane no bounded follower command can absorb a
                # ~5 cm leader deviation (thousands of N); see the report.
                eps = eps+leader_eps[None, :]*np.abs(g[..., 6:]).sum(axis=-1)
            target = np.exp(-alpha*decay_t)[None, :]*y0[:, None]+eps
            a = g[..., :6]
            b = target-y+np.einsum('rnc,c->rn', a, u_ref.ravel())
            rows.append(a.reshape(-1, 6)/scales[kind])
            rhs.append(b.ravel()/scales[kind])
            groups.append(np.full(b.size, group))
            kinds.append(np.full(b.size, kind))
            records.append((kind, keys, y, g))
            if kind == CLEARANCE:
                # Terminal look-ahead: still safe after tau at the predicted
                # approach rate (a braking-margin terminal set).
                decay = np.exp(-alpha*(decay_t[-1]+cfg.terminal_lookahead))
                rows.append(g_term[:, :6]/scales[kind])
                rhs.append((decay*y0+eps[:, -1]-y_term+g_term[:, :6]@u_ref.ravel())/scales[kind])
                groups.append(np.zeros(len(idx), dtype=int))
                kinds.append(np.full(len(idx), 'terminal'))
        rows = np.concatenate(rows) if rows else np.empty((0, 6))
        rhs = np.concatenate(rhs) if rhs else np.empty(0)
        groups = np.concatenate(groups) if groups else np.empty(0, dtype=int)
        kinds = np.concatenate(kinds) if kinds else np.empty(0, dtype=str)
        # A row is redundant if it holds for every command in the rate box.
        worst = (np.minimum(rows*(lower-u_ref.ravel()), rows*(upper-u_ref.ravel()))).sum(axis=1)
        keep = rows@u_ref.ravel()+worst < rhs
        timing['rows_ms'] = 1000*(time.perf_counter()-begin)

        begin = time.perf_counter()
        center = (nominal.ravel()+cfg.command_smoothing*u_ref.ravel())/(1+cfg.command_smoothing)
        u, status, slacks, violation = safety_qp(center, rows[keep], rhs[keep], groups[keep], lower, upper)
        timing['qp_ms'] = 1000*(time.perf_counter()-begin)
        command = u.reshape(2, 3)

        # Store predictions under the chosen command for residual scoring.
        applied = np.concatenate((u, leaders.ravel()))
        delta = u-u_ref.ravel()
        predicted_min = None  # JSON null when no clearance rows (Infinity is not valid JSON)
        for kind, keys, y, g in records:
            y_u = y+np.einsum('rnc,c->rn', g[..., :6], delta)
            if kind == CLEARANCE:
                predicted_min = float(y_u.min())
            for k in range(self.outputs):
                arrival = self.frame+self.stride*(k+1)
                for r, key in enumerate(keys):
                    self.pending.setdefault(arrival, []).append(
                        (kind, int(key), k, self.frame, float(y_u[r, k]), g[r, k], applied))
        first = out['q'][0]+np.einsum('mxc,c->mx', out['sq'][0][..., :6], delta)
        self.observed.append((self.frame+self.stride, self.frame, first,
                              out['sq'][0].reshape(-1, 12), applied))
        self.history[self.frame] = applied
        self.history.pop(self.frame-4*cfg.horizon_frames, None)
        self.u_prev = command.copy()
        self.frame += 1

        eps_now = {kind: self.conformal[kind].bounds().tolist() for kind in self.conformal}
        residual = rhs-rows@u
        worst = {str(kind): float(residual[kinds == kind].max()) for kind in np.unique(kinds)}
        debug = dict(status=status, mode='sensitivity_cbf', **state_debug,
                     min_clearance=float(d.min()), h_min=float(h_now.min()),
                     initially_safe=bool(h_now.min() >= 0), candidate_count=int(len(idx)),
                     total_rows=int(keep.sum()), generated_rows=int(len(rhs)),
                     predicted_min_clearance_barrier=predicted_min,
                     horizon_min_clearance_barrier=float(h_min.min()),
                     max_violation=violation, slack_soft=float(slacks[0]), slack_hard=float(slacks[1]),
                     worst_row_residual=worst,
                     correction=float(np.linalg.norm(command-nominal)),
                     tension=tension_now.tolist(), strain_ks=strain_now.tolist(),
                     bias_norm=float(np.linalg.norm(self.bias)) if self.bias is not None else 0.,
                     conformal=eps_now, controller_ms=1000*(time.perf_counter()-started), **timing,
                     follower_velocity=command.tolist(), nominal_velocity=nominal.tolist(),
                     candidate_indices=idx.tolist(), detected_indices=idx.tolist(), basin_indices=[],
                     surfaces=surfaces[idx], rows=rows[keep], rhs=rhs[keep],
                     safety_scope=('Linearized coarse prediction with conformal tightening; '
                                   'not a certificate of whole-mesh safety'))
        debug.update(red.info)
        return command, debug

"""Pure NumPy/SciPy geometry and velocity filter; no Isaac or ROS imports.

GID-inspired, not a certified cloth CBF: material-coordinate interpolation is
an approximate deformation Jacobian and basin averaging is approximate too.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class Settings:
    sectors: int = 24
    bands: int = 18
    margin: float = 0.05
    activation: float = 0.12  # clearance beyond margin
    epsilon_min: float = 0.001
    epsilon_band: float = 0.015
    horizon: int = 10
    local_radius: int = 2
    sigma: float = 1.25
    alpha: float = 2.0
    max_speed: float = 0.0  # zero disables the optional follower speed bound
    cell_guards: bool = True
    stretch_ratio: float = 1.02
    formation_gain: float = 0.5
    predictive_enabled: bool = True
    ray_basins_enabled: bool = True
    strain_enabled: bool = False
    vertical_formation_recovery: bool = False
    basin_knn: int = 4
    basin_neighbor_radius: float = 0.08
    weft_strain_limit: float = 0.02  # policy limits, not measured failure strains
    warp_strain_limit: float = 0.02
    shear_limit: float = 0.20  # absolute cosine of material-axis angle
    strain_activation: float = 0.005
    sensitivity_step: float = 0.02  # m/s; ~0.33 mm/frame at 60 Hz
    sensitivity_refresh_frames: int = 5
    sensitivity_state_threshold: float = 0.003  # maximum particle motion, m
    sensitivity_command_threshold: float = 0.02  # m/s
    prediction_iterations: int = 2
    prediction_trust_displacement: float = 0.005  # per-axis change per iteration
    clearance_tightening: float = 0.0001
    strain_tightening: float = 0.00001

    def __post_init__(self):
        for name in ('sectors', 'bands', 'horizon', 'local_radius',
                     'sensitivity_refresh_frames', 'prediction_iterations', 'basin_knn'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.sectors < 4 or self.bands < 4:
            raise ValueError('Atlas requires at least four sectors and bands')
        for name in ('margin', 'activation', 'epsilon_min', 'epsilon_band',
                     'sigma', 'alpha', 'max_speed', 'stretch_ratio', 'formation_gain',
                     'weft_strain_limit', 'warp_strain_limit', 'shear_limit',
                     'strain_activation', 'sensitivity_step', 'sensitivity_state_threshold',
                     'sensitivity_command_threshold', 'prediction_trust_displacement',
                     'clearance_tightening', 'strain_tightening', 'basin_neighbor_radius'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {name}')
        if self.sigma == 0 or self.stretch_ratio < 1:
            raise ValueError('sigma must be positive; stretch_ratio >= 1')
        if (self.sensitivity_step == 0 or self.prediction_trust_displacement == 0
                or min(self.weft_strain_limit, self.warp_strain_limit, self.shear_limit) <= 0
                or self.shear_limit >= 1):
            raise ValueError('Invalid prediction step, trust displacement or strain limits')


class Capsule:
    """Z-axis capsule enclosing the entire axis-aligned box, including corners."""
    def __init__(self, center, half_extents, sectors=24, bands=18):
        self.center = np.asarray(center, dtype=float)
        half = np.asarray(half_extents, dtype=float)
        if (self.center.shape != (3,) or half.shape != (3,)
                or not np.isfinite(self.center).all() or not np.isfinite(half).all()
                or np.any(half <= 0)):
            raise ValueError('Expected finite center and positive box half extents')
        self.radius = float(np.linalg.norm(half[:2]))
        self.half_length = float(half[2])
        self.sectors, self.bands = sectors, bands
        self.meridian_length = np.pi * self.radius + 2 * self.half_length
        # Centers of a periodic azimuth x meridian surface atlas.
        self.cells = self.surface((np.arange(bands) + .5) / bands,
                                  (np.arange(sectors) + .5) * 2 * np.pi / sectors)
        self.tree = cKDTree(self.cells)
        self.neighbors = [self._neighbors(i) for i in range(len(self.cells))]

    def surface(self, fractions, angles):
        s = np.asarray(fractions) * self.meridian_length
        cap = np.pi * self.radius / 2
        theta_lo = s / self.radius
        theta_hi = (s - cap - 2 * self.half_length) / self.radius
        rho = np.where(s < cap, self.radius * np.sin(theta_lo),
                       np.where(s > cap + 2 * self.half_length,
                                self.radius * np.cos(theta_hi), self.radius))
        z = np.where(s < cap, -self.half_length - self.radius * np.cos(theta_lo),
                     np.where(s > cap + 2 * self.half_length,
                              self.half_length + self.radius * np.sin(theta_hi),
                              -self.half_length + s - cap))
        a = np.asarray(angles)
        return (np.stack(np.broadcast_arrays(rho[:, None] * np.cos(a),
                                             rho[:, None] * np.sin(a), z[:, None]), axis=-1)
                .reshape(-1, 3) + self.center)

    def _neighbors(self, cell):
        b, a = divmod(cell, self.sectors)
        out = [b * self.sectors + (a + d) % self.sectors for d in (-1, 1)]
        out += [bb * self.sectors + a for bb in (b - 1, b + 1) if 0 <= bb < self.bands]
        return out

    def distances(self, points, particle_radius=0.):
        p = np.asarray(points, dtype=float)
        axis = np.broadcast_to(self.center, p.shape).copy()
        axis[:, 2] = np.clip(p[:, 2], self.center[2] - self.half_length,
                            self.center[2] + self.half_length)
        delta = p - axis
        lengths = np.linalg.norm(delta, axis=1)
        normals = delta / np.maximum(lengths[:, None], 1e-12)
        normals[lengths < 1e-12] = [1., 0., 0.]  # deterministic subgradient on axis
        surface = axis + self.radius * normals
        return lengths - self.radius - particle_radius, normals, surface

    def first_visible(self, cell, visible, horizon):
        b, a = divmod(cell, self.sectors)
        found = []
        for db, da in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for step in range(1, horizon + 1):
                bb = b + db * step
                if not 0 <= bb < self.bands:
                    break
                cc = bb * self.sectors + (a + da * step) % self.sectors
                if cc == cell:
                    break
                if cc in visible:
                    found.append(cc)
                    break
        return found


def material_weights(points):
    """Fixed rest-XY bilinear influence, ordered FL, FR, LL, LR."""
    xy = np.asarray(points)[:, :2]
    span = np.ptp(xy, axis=0)
    if np.any(span <= 0):
        raise ValueError('Rest cloth must span both X and Y')
    x, y = ((xy - xy.min(axis=0)) / span).T
    return np.column_stack(((1-x)*(1-y), x*(1-y), (1-x)*y, x*y))


def solve_filter(nominal, rows, rhs, max_speed=0.0, bounds=None):
    """Convex quadratic objective with simultaneous rows and optional speed balls.

    Zero max_speed disables speed limiting, including during violation recovery.
    Infeasible/failed solves use least-squares violation recovery and report
    residuals. Recovery is explicitly not reported as a safe solution.
    """
    nominal = np.asarray(nominal, dtype=float).reshape(6)
    A = np.asarray(rows, dtype=float).reshape(-1, 6)
    b = np.asarray(rhs, dtype=float)
    def balls(u):
        return max_speed**2 - np.sum(u.reshape(2, 3)**2, axis=1)
    def balls_jac(u):
        jac = np.zeros((2, 6))
        jac[0, :3], jac[1, 3:] = -2*u[:3], -2*u[3:]
        return jac
    start = nominal.reshape(2, 3).copy()
    if max_speed > 0:
        start *= np.minimum(1., max_speed / np.maximum(np.linalg.norm(start, axis=1), 1e-12))[:, None]
    start = start.ravel()
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float).reshape(6, 2)
        start = np.clip(start, bounds[:, 0], bounds[:, 1])
    speed_constraints = ([{'type': 'ineq', 'fun': balls, 'jac': balls_jac}]
                         if max_speed > 0 else [])
    def product(u):
        # Six-column matrix-vector products do not benefit from a large BLAS
        # thread pool; einsum avoids thread-launch overhead on every callback.
        return np.einsum('ij,j->i', A, u, optimize=False)
    # SLSQP's least-squares subproblem becomes extremely slow with tens of
    # thousands of inequalities. Constraint generation keeps its working set
    # small, but checks ALL inequalities before accepting any command.
    if len(b) > 128:
        active = set(np.argsort(b-product(start))[-16:].tolist())
        rounds = 8
    else:
        active = set(range(len(b)))
        rounds = 1
    trial_start = start
    for _ in range(rounds):
        indices = np.array(sorted(active), dtype=int)
        local_A, local_b = A[indices], b[indices]
        constraints = list(speed_constraints)
        if len(indices):
            constraints.append({'type': 'ineq',
                                'fun': lambda u: np.einsum('ij,j->i', local_A, u, optimize=False)-local_b,
                                'jac': lambda u: local_A})
        result = minimize(lambda u: .5 * np.sum((u-nominal)**2), trial_start,
                          jac=lambda u: u-nominal, constraints=constraints,
                          bounds=bounds, method='SLSQP', options={'maxiter': 80, 'ftol': 1e-10})
        if not result.success or not np.isfinite(result.x).all():
            break
        violations = b-product(result.x)
        if not len(b) or np.max(violations) <= 1e-6:
            break
        candidates = np.flatnonzero(violations > 1e-6)
        added = set(candidates[np.argsort(violations[candidates])[-16:]].tolist())-active
        if not added:
            break
        active.update(added)
        trial_start = result.x
    u = result.x
    valid = (result.success and np.isfinite(u).all()
             and (max_speed <= 0 or np.min(balls(u)) >= -1e-9)
             and (bounds is None or (np.all(u >= bounds[:, 0]-1e-9)
                                    and np.all(u <= bounds[:, 1]+1e-9)))
             and (not len(b) or np.min(product(u) - b) >= -1e-6))
    status = 'feasible'
    if not valid:
        def objective(u):
            v = np.maximum(b - product(u), 0.)
            return .5 * np.sum((u-nominal)**2) + 5000. * (v @ v)
        def gradient(u):
            return u-nominal - 10000. * np.einsum(
                'ij,i->j', A, np.maximum(b-product(u), 0.), optimize=False)
        # With only box bounds, L-BFGS-B handles the steep violation penalty
        # more reliably than an infeasible SLSQP linear-constraint solve.
        recovery = minimize(objective, start, jac=gradient, constraints=speed_constraints,
                            bounds=bounds, method='SLSQP' if speed_constraints else 'L-BFGS-B',
                            options={'maxiter': 80, 'ftol': 1e-12})
        u = recovery.x if np.isfinite(recovery.x).all() else np.zeros(6)
        status = 'recovery_not_safe'
    # Remove numerical speed overshoot, then report actual post-limit residual.
    v = u.reshape(2, 3)
    if max_speed > 0:
        v *= np.minimum(1., max_speed / np.maximum(np.linalg.norm(v, axis=1), 1e-12))[:, None]
    violation = float(np.max(np.maximum(b - product(u), 0.))) if len(b) else 0.
    if violation > 1e-6:
        status = 'recovery_not_safe'
    return u.reshape(2, 3), status, violation


class CapsuleController:
    def __init__(self, capsule, rest_points, rest_corners, particle_radius, settings=None):
        self.capsule, self.settings = capsule, settings or Settings()
        self.weights = material_weights(rest_points)
        self.rest_corners = np.asarray(rest_corners, dtype=float).copy()
        self.particle_radius = particle_radius
        self.previous_points = None
        self.previous_commands = np.zeros((4, 3))

    def basins(self, cell_distances):
        c, cfg = self.capsule, self.settings
        visible = set(cell_distances)
        seeds = {i for i in visible if all(cell_distances[i] <= cell_distances[j] + cfg.epsilon_min
                 for j in c.first_visible(i, visible, cfg.horizon))}
        support = set()
        for seed in sorted(seeds):
            todo, seen = [seed], {seed}
            while todo:
                i = todo.pop()
                support.add(i)
                for j in c.neighbors[i]:
                    if (j in visible and j not in seen
                            and abs(cell_distances[j]-cell_distances[seed]) <= cfg.epsilon_band):
                        seen.add(j)
                        todo.append(j)
        groups = []
        while support:
            root = min(support)
            group, todo = {root}, [root]
            support.remove(root)
            while todo:
                i = todo.pop()
                for j in c.neighbors[i]:
                    if j in support and abs(cell_distances[j]-cell_distances[i]) <= cfg.epsilon_band:
                        support.remove(j)
                        group.add(j)
                        todo.append(j)
            if group & seeds:
                groups.append((sorted(group), sorted(group & seeds)))
        return groups

    def update(self, points, corners, leaders, dt):
        cfg, cap = self.settings, self.capsule
        points, corners, leaders = map(lambda x: np.asarray(x, dtype=float), (points, corners, leaders))
        if (points.shape != (len(self.weights), 3) or corners.shape != (4, 3)
                or leaders.shape != (2, 3) or not all(np.isfinite(x).all() for x in (points, corners, leaders))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Invalid controller state')
        d, normals, surfaces = cap.distances(points, self.particle_radius)
        ids = cap.tree.query(surfaces)[1]
        # Every particle is assigned; empty cells remain invisible, not zero distance.
        retained = {}
        for i in np.argsort(d, kind='stable'):
            retained.setdefault(int(ids[i]), int(i))
        distances = {cell: float(d[i]) for cell, i in retained.items()}
        drift = np.zeros_like(points)
        if self.previous_points is not None:
            drift = (points-self.previous_points)/dt - self.weights @ self.previous_commands
        known = self.weights[:, 2:] @ leaders + drift
        point_rows = (self.weights[:, :2, None] * normals[:, None, :]).reshape(-1, 6)
        point_drift = np.einsum('ij,ij->i', normals, known)
        rows, rhs, basin_points = [], [], []
        groups = self.basins(distances)
        for cells, seeds in groups:
            h = min(distances[s] for s in seeds) - cfg.margin
            if h > cfg.activation:
                continue
            local_rows, local_drifts = [], []
            for seed in seeds:
                sb, sa = divmod(seed, cap.sectors)
                cb, ca = np.divmod(cells, cap.sectors)
                da = np.minimum(abs(ca-sa), cap.sectors-abs(ca-sa))
                radius2 = (cb-sb)**2 + da**2
                chosen = np.flatnonzero(radius2 <= cfg.local_radius**2)
                weights = np.exp(-radius2[chosen]/(2*cfg.sigma**2))
                weights /= weights.sum()
                idx = [retained[cells[j]] for j in chosen]
                local_rows.append(weights @ point_rows[idx])
                local_drifts.append(weights @ point_drift[idx])
            rows.append(np.mean(local_rows, axis=0))
            rhs.append(-cfg.alpha*h - np.mean(local_drifts))
            basin_points.append(retained[min(seeds, key=distances.get)])
        basin_count = len(rows)
        active = [i for i in retained.values() if d[i]-cfg.margin <= cfg.activation]
        if cfg.cell_guards:
            for i in active:
                rows.append(point_rows[i])
                rhs.append(-cfg.alpha*(d[i]-cfg.margin)-point_drift[i])
        # Upper stretch barriers, with both followers in the SAME edge row.
        for i, j in (((0, 2), (1, 3), (0, 1)) if cfg.strain_enabled else ()):
            delta = corners[j]-corners[i]
            length = np.linalg.norm(delta)
            direction = delta / max(length, 1e-12)
            h = cfg.stretch_ratio*np.linalg.norm(self.rest_corners[j]-self.rest_corners[i])-length
            row = np.zeros((2, 3))
            row[i] += direction
            known_stretch = 0.
            if j < 2:
                row[j] -= direction
            else:
                known_stretch = float(direction @ leaders[j-2])
            rows.append(row.ravel())
            rhs.append(-cfg.alpha*h + known_stretch)
        offset = self.rest_corners[:2]-self.rest_corners[2:]
        error = corners[2:]+offset-corners[:2]
        if not cfg.vertical_formation_recovery:
            error[:, 2] = 0
        nominal = leaders + cfg.formation_gain*error
        command, status, violation = solve_filter(nominal, rows, rhs, cfg.max_speed)
        self.previous_points = points.copy()
        self.previous_commands = np.vstack((command, leaders))
        debug = dict(status=status, min_clearance=float(d.min()), h_min=float(d.min()-cfg.margin),
                     basin_rows=basin_count, total_rows=len(rows), occupied_cells=len(retained),
                     max_violation=violation, correction=float(np.linalg.norm(command-nominal)),
                     detected_indices=active, basin_indices=basin_points,
                     surfaces=surfaces[active], rows=np.asarray(rows), rhs=np.asarray(rhs))
        return command, debug

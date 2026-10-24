"""Shared full-cloth Newton transition for live stepping and isolated rollouts.

Only immutable model data is shared. Every runtime owns states, contacts,
collision pipeline, Style3D workspaces, boundary arrays and CUDA graphs.
"""
import numpy as np
import time
import newton
import warp as wp


@wp.func
def boundary_geometry(p: wp.vec3, center: wp.vec3, h: float, r: float, centered_top: int):
    if centered_top == 0:
        a = wp.vec3(center[0], center[1], wp.clamp(p[2], center[2]-h, center[2]+h))
        length = wp.length(p-a)
        normal = wp.vec3(1.0, 0.0, 0.0)
        if length > 1.e-12:
            normal = (p-a)/length
        return length-r, normal
    delta = p-center
    rho = wp.length(wp.vec2(delta[0], delta[1]))
    z = delta[2]
    radius = wp.sqrt(r*r+h*h)
    profile = wp.vec2(rho, z)
    bottom = wp.vec2(r, -h)
    if z < -h:
        direction = wp.vec2(rho, z+h)
        bottom = wp.vec2(0.0, -h)+r*direction/wp.max(wp.length(direction), 1.e-12)
    side = wp.vec2(r, wp.clamp(z, -h, h))
    top = wp.vec2(r, h)
    length_top = wp.length(profile)
    if z/wp.max(length_top, 1.e-12) >= h/radius:
        top = radius*profile/wp.max(length_top, 1.e-12)
    nearest = bottom
    surface_normal = wp.vec2(bottom[0]/r, (bottom[1]+h)/r)
    distance = wp.length(profile-bottom)
    if wp.length(profile-side) < distance:
        distance = wp.length(profile-side)
        nearest = side
        surface_normal = wp.vec2(1.0, 0.0)
    if wp.length(profile-top) < distance:
        distance = wp.length(profile-top)
        nearest = top
        surface_normal = top/radius
    sign = float(1.0)
    if (rho <= r and z >= -h-wp.sqrt(wp.max(r*r-rho*rho, 0.0))
            and z <= wp.sqrt(wp.max(radius*radius-rho*rho, 0.0))):
        sign = -1.0
    normal2 = surface_normal
    if distance > 1.e-12:
        normal2 = sign*(profile-nearest)/distance
    radial = wp.vec2(1.0, 0.0)
    if rho > 1.e-12:
        radial = wp.vec2(delta[0], delta[1])/rho
    return sign*distance, wp.vec3(normal2[0]*radial[0], normal2[0]*radial[1], normal2[1])


@wp.func
def boundary_distance(p: wp.vec3, center: wp.vec3, h: float, r: float, centered_top: int):
    distance, normal = boundary_geometry(p, center, h, r, centered_top)
    return distance


@wp.kernel
def cast_capsule_rays(mesh: wp.uint64, center: wp.vec3,
                      cells: wp.array[wp.vec3], faces: wp.array[int],
                      weights: wp.array[wp.vec3], points: wp.array[wp.vec3]):
    i = wp.tid()
    direction = wp.normalize(cells[i]-center)
    query = wp.mesh_query_ray(mesh, center, direction, 1.e6)
    faces[i] = -1
    if query.result:
        faces[i] = query.face
        weights[i] = wp.vec3(query.u, query.v, 1.-query.u-query.v)
        points[i] = center+query.t*direction


@wp.kernel
def observe_hits(q: wp.array[wp.vec3], triangles: wp.array2d[int],
                 faces: wp.array[int], weights: wp.array[wp.vec3],
                 offsets: wp.array[wp.vec3], count: wp.array[int], center: wp.vec3, half_length: float,
                 radius: float, centered_top: int, particle_radius: float, margin: float,
                 step: int, out: wp.array2d[float]):
    i = wp.tid()
    if i < count[0]:
        f, w = faces[i], weights[i]
        p = offsets[i]+w[0]*q[triangles[f, 0]]+w[1]*q[triangles[f, 1]]+w[2]*q[triangles[f, 2]]
        out[step, i] = boundary_distance(p, center, half_length, radius, centered_top)-particle_radius-margin


class CapsuleRayDetector:
    """BVH refit follows canonical live positions; rays do not mutate them."""
    def __init__(self, model, live_state, capsule):
        self.device, self.capsule = model.device, capsule
        self.indices = wp.array(model.tri_indices.numpy().ravel(), dtype=int, device=self.device)
        self.mesh = wp.Mesh(points=live_state.particle_q, indices=self.indices)
        self.cells = wp.array(capsule.cells, dtype=wp.vec3, device=self.device)
        self.faces = wp.empty(len(capsule.cells), dtype=int, device=self.device)
        self.weights = wp.empty(len(capsule.cells), dtype=wp.vec3, device=self.device)
        self.points = wp.empty(len(capsule.cells), dtype=wp.vec3, device=self.device)

    def detect(self):
        self.mesh.refit()
        wp.launch(cast_capsule_rays, len(self.cells),
                  inputs=[self.mesh.id, wp.vec3(*self.capsule.center), self.cells,
                          self.faces, self.weights, self.points], device=self.device)
        faces = self.faces.numpy()
        valid = faces >= 0
        return dict(faces=faces[valid], weights=self.weights.numpy()[valid],
                    points=self.points.numpy()[valid])


@wp.kernel
def move_boundary(indices: wp.array[int], corner_ids: wp.array[int],
                  offsets: wp.array[wp.vec3], starts: wp.array[wp.vec3],
                  commands: wp.array[wp.vec3], elapsed: float,
                  q: wp.array[wp.vec3], qd: wp.array[wp.vec3]):
    i = wp.tid()
    c, j = corner_ids[i], indices[i]
    q[j] = starts[c] + elapsed*commands[c] + offsets[i]
    qd[j] = commands[c]


@wp.kernel
def observe_particles(q: wp.array[wp.vec3], start: int, count: int,
                      center: wp.vec3, half_length: float, radius: float,
                      centered_top: int, particle_radius: float, margin: float, step: int,
                      out: wp.array2d[float]):
    i = wp.tid()
    p = q[start+i]
    out[step, i] = boundary_distance(p, center, half_length, radius, centered_top)-particle_radius-margin


@wp.kernel
def observe_triangles(q: wp.array[wp.vec3], triangles: wp.array2d[int],
                      rest: wp.array[wp.mat22], count: int,
                      weft_limit: float, warp_limit: float, shear_limit: float,
                      step: int, out: wp.array2d[float]):
    i = wp.tid()
    a, b, c = triangles[i, 0], triangles[i, 1], triangles[i, 2]
    dm = rest[i]
    e1, e2 = q[b]-q[a], q[c]-q[a]
    fu = e1*dm[0, 0]+e2*dm[1, 0]
    fv = e1*dm[0, 1]+e2*dm[1, 1]
    lu, lv = wp.length(fu), wp.length(fv)
    shear = wp.dot(fu, fv)/wp.max(lu*lv, 1.e-12)
    j = count+4*i
    out[step, j] = 1.+weft_limit-lu
    out[step, j+1] = 1.+warp_limit-lv
    out[step, j+2] = shear_limit-shear
    out[step, j+3] = shear_limit+shear


class ClothRuntime:
    def __init__(self, model, patch_indices, corner_ids, offsets, substeps, dt,
                 iterations, linear_iterations, *, solver=None, capsule=None,
                 particle_start=0, particle_count=None, settings=None,
                 particle_radius=0., capture=True):
        if substeps < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive substeps and timestep required')
        self.model, self.device = model, model.device
        self.substeps, self.dt = substeps, dt
        self.solver = solver or newton.solvers.SolverStyle3D(
            model, iterations=iterations, linear_iterations=linear_iterations)
        # Never use model.contacts(pipeline) or model.collide(pipeline): those
        # replace the pipeline cached on the shared Model.
        self.pipeline = newton.CollisionPipeline(model, broad_phase='explicit',
                                                soft_contact_margin=model.soft_contact_margin)
        self.contacts = self.pipeline.contacts()
        self.state = model.state()
        self.scratch = model.state()
        self.control = model.control()
        self.indices = wp.array(patch_indices, dtype=int, device=self.device)
        self.corner_ids = wp.array(corner_ids, dtype=int, device=self.device)
        self.offsets = wp.array(offsets, dtype=wp.vec3, device=self.device)
        self.starts = wp.zeros(4, dtype=wp.vec3, device=self.device)
        self.commands = wp.zeros(4, dtype=wp.vec3, device=self.device)
        starts = np.zeros((4, 3), dtype=np.float32)
        rest = model.particle_q.numpy()
        for index, corner, offset in zip(patch_indices, corner_ids, offsets):
            starts[int(corner)] = rest[int(index)]-offset
        self.starts.assign(starts)
        self.capsule, self.settings = capsule, settings
        self.particle_radius = particle_radius
        self.particle_start = particle_start
        self.particle_count = particle_count or model.particle_count
        self.observations = None
        self.hit_observations = None
        if capsule is not None:
            self.observations = wp.zeros((substeps, self.particle_count+(4*model.tri_count if settings.strain_enabled else 0)),
                                         dtype=float, device=self.device)
            capacity = len(capsule.cells)
            self.hit_faces = wp.zeros(capacity, dtype=int, device=self.device)
            self.hit_weights = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
            self.hit_offsets = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
            self.hit_count = wp.zeros(1, dtype=int, device=self.device)
            self.hit_observations = wp.zeros((substeps, capacity), dtype=float, device=self.device)
        self.graph = None
        if capsule is not None and hasattr(self.solver, 'configure_observations'):
            self.solver.configure_observations(self)
        if capture and wp.get_device(self.device).is_cuda:
            saved = model.state()
            saved.assign(self.state)
            # Warm compile, allocation and collision kernels before capture.
            self._advance()
            self.state.assign(saved)
            self.scratch.assign(saved)
            with wp.ScopedCapture(device=self.device) as scope:
                self._advance()
            self.graph = scope.graph
            self.state.assign(saved)
            self.scratch.assign(saved)

    def _pin(self, state, elapsed):
        wp.launch(move_boundary, len(self.indices),
                  inputs=[self.indices, self.corner_ids, self.offsets, self.starts,
                          self.commands, elapsed, state.particle_q, state.particle_qd],
                  device=self.device)

    def _observe(self, state, step):
        if self.observations is None:
            return
        c, cfg = self.capsule, self.settings
        wp.launch(observe_particles, self.particle_count,
                  inputs=[state.particle_q, self.particle_start, self.particle_count,
                          wp.vec3(*c.center), c.half_length, c.radius,
                          int(c.centered_top), self.particle_radius, cfg.margin, step, self.observations],
                  device=self.device)
        if cfg.strain_enabled:
            wp.launch(observe_triangles, self.model.tri_count,
                      inputs=[state.particle_q, self.model.tri_indices, self.model.tri_poses,
                              self.particle_count, cfg.weft_strain_limit, cfg.warp_strain_limit,
                              cfg.shear_limit, step, self.observations], device=self.device)
        wp.launch(observe_hits, len(self.hit_faces),
                  inputs=[state.particle_q, self.model.tri_indices, self.hit_faces,
                          self.hit_weights, self.hit_offsets, self.hit_count, wp.vec3(*c.center),
                          c.half_length, c.radius, int(c.centered_top), self.particle_radius, cfg.margin,
                          step, self.hit_observations], device=self.device)
        if hasattr(self.solver, 'observe_tangent'):
            self.solver.observe_tangent(self, state, step)

    def _advance(self):
        if hasattr(self.solver, 'reset_trajectory'):
            self.solver.reset_trajectory()
        a, b = self.state, self.scratch
        for k in range(self.substeps):
            a.clear_forces()
            # Backward Euler evaluates prescribed boundaries at the end of
            # each interval; inactive pin DOFs stay fixed inside the solve.
            elapsed = (k+1)*self.dt
            self._pin(a, elapsed)
            self.pipeline.collide(a, self.contacts)
            self.solver.step(a, b, self.control, self.contacts, self.dt)
            self._pin(b, elapsed)
            self._observe(b, k)
            a, b = b, a
        # Canonical output buffer also works for odd substep counts in graphs.
        if a is not self.state:
            self.state.assign(a)

    def advance(self, starts, commands):
        starts, commands = np.asarray(starts), np.asarray(commands)
        if (starts.shape != (4, 3) or commands.shape != (4, 3) or
                not np.isfinite(starts).all() or not np.isfinite(commands).all()):
            raise ValueError('Invalid boundary state or commands')
        self.starts.assign(starts.astype(np.float32))
        self.commands.assign(commands.astype(np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._advance()


class NewtonPredictor:
    def __init__(self, model, patch_indices, corner_ids, offsets, substeps, dt,
                 iterations, linear_iterations, capsule, particle_radius,
                 settings, *, particle_start=0, particle_count=None, capture=True):
        end = particle_start+(particle_count or model.particle_count)
        triangles = model.tri_indices.numpy()
        if (model.body_count or not len(triangles) or np.any(triangles < particle_start)
                or np.any(triangles >= end)):
            raise ValueError('Predictor requires static obstacles and a single cloth triangle mesh')
        self.tangent = None
        self.sensitivity_backend = settings.sensitivity_backend
        solver = None
        if settings.sensitivity_method != 'finite_difference':
            if settings.sensitivity_backend == 'warp':
                from .warp_sensitivity import TangentStyle3D
                solver = TangentStyle3D(model, patch_indices, corner_ids,
                    iterations=iterations, linear_iterations=linear_iterations)
                self.tangent = solver
            else:
                from .sensitivity_solver import SensitivityStyle3D
                from .dynamics_sensitivity import ClothDynamicsSensitivity
                solver = SensitivityStyle3D(model, iterations=iterations,
                                            linear_iterations=linear_iterations, substeps=substeps)
                self.tangent = ClothDynamicsSensitivity(model, patch_indices, corner_ids)
        self.runtime = ClothRuntime(model, patch_indices, corner_ids, offsets,
                                    substeps, dt, iterations, linear_iterations,
                                    solver=solver,
                                    capsule=capsule, settings=settings,
                                    particle_start=particle_start,
                                    particle_count=particle_count,
                                    particle_radius=particle_radius, capture=capture)
        self.snapshot = model.state()
        self.starts = None
        self.calls = 0
        self.progress_callback = None
        self.last_progress = 0.
        self.selected_hit_count = None
        self.last_commands = None
        self.last_sensitivity_timings = {}

    def select_hits(self, faces, weights):
        faces, weights = np.asarray(faces), np.asarray(weights)
        r = self.runtime
        n, capacity = len(faces), len(r.hit_faces)
        if (n > capacity or weights.shape != (n, 3) or np.any(faces < 0)
                or np.any(faces >= r.model.tri_count) or not np.isfinite(weights).all()
                or np.any(weights < -1e-6) or not np.allclose(weights.sum(axis=1), 1., atol=1e-6)):
            raise ValueError('Invalid selected material hits')
        f = np.zeros(capacity, dtype=np.int32)
        w = np.zeros((capacity, 3), dtype=np.float32)
        f[:n], w[:n] = faces, weights
        r.hit_faces.assign(f)
        r.hit_weights.assign(w)
        r.hit_count.assign(np.array([n], dtype=np.int32))
        self.selected_hit_count = n
        self.selected_faces, self.selected_weights = faces.copy(), weights.copy()

    def begin(self, live_state, starts):
        self.snapshot.assign(live_state)
        self.starts = np.asarray(starts, dtype=np.float32).copy()
        self.calls = 0
        self.last_commands = None
        self.last_sensitivity_timings = {}

    def current_cloth_state(self):
        """Full q/v state from the immutable frame-start prediction snapshot."""
        if self.starts is None:
            raise RuntimeError('Call begin before reading predictor state')
        from .cloth_state import ClothState
        return ClothState.from_newton(self.snapshot, self.runtime.particle_start,
                                      self.runtime.particle_count)

    def __call__(self, followers, leaders):
        if self.starts is None:
            raise RuntimeError('Call begin before predicting')
        r = self.runtime
        if self.tangent is not None:
            # Graph replay does not execute Python reset hooks.
            r.solver.invalidate_host()
        r.state.assign(self.snapshot)
        r.scratch.assign(self.snapshot)
        r.advance(self.starts, np.vstack((followers, leaders)))
        self.last_commands = np.vstack((followers, leaders)).copy()
        self.calls += 1
        observations = (r.observations.numpy().astype(float) if self.selected_hit_count is None
                        else r.hit_observations.numpy()[:, :self.selected_hit_count].astype(float))
        # Isaac's main thread also owns its UI. Pump Kit between expensive
        # rollouts without advancing the live Newton state or changing the
        # leader command held fixed for this control frame.
        self._progress()
        return observations

    def _progress(self):
        now = time.perf_counter()
        if self.progress_callback is not None and now-self.last_progress >= .05:
            self.progress_callback()
            self.last_progress = time.perf_counter()

    def sensitivity(self, followers, leaders):
        """Differentiate the last nominal trajectory without another rollout."""
        from .dynamics_sensitivity import SensitivityUnavailable, observation_jacobian
        if self.tangent is None:
            raise SensitivityUnavailable('solver tangent not enabled')
        if self.last_commands is None or not np.array_equal(
                self.last_commands, np.vstack((followers, leaders))):
            raise SensitivityUnavailable('nominal trajectory does not match commands')
        r = self.runtime
        if r.solver.contact_active:
            raise SensitivityUnavailable('contact active in nominal trajectory')
        if self.sensitivity_backend == 'warp':
            if r.solver.invalid_flag.numpy()[0]:
                raise SensitivityUnavailable('degenerate material direction')
            started = time.perf_counter()
            result = (r.solver.jacobian.numpy() if self.selected_hit_count is None
                      else r.solver.hit_jacobian.numpy()[:, :self.selected_hit_count])
            if not np.isfinite(result).all():
                raise SensitivityUnavailable('nonfinite device tangent')
            self.last_sensitivity_timings = dict(backend='warp',
                transfer_ms=1000*(time.perf_counter()-started), replay_ms=0.,
                predictor_graph_enabled=r.graph is not None,
                derivative_compute_in_rollout=True)
            return result.astype(float)
        transfer_started = time.perf_counter()
        positions = r.solver.positions
        # Materialize the six batched host arrays once before the replay timer.
        r.solver.linear_records
        transfer_ms = 1000*(time.perf_counter()-transfer_started)
        self._progress()
        if len(positions) != r.substeps:
            raise SensitivityUnavailable('incomplete nominal trajectory')
        faces = self.selected_faces if self.selected_hit_count is not None else None
        weights = self.selected_weights if faces is not None else None
        replay_started = time.perf_counter()
        tangents = self.tangent.solver_trajectory(r.solver, r.dt, progress=self._progress)
        result = np.stack([observation_jacobian(q, s, r, faces, weights)
                           for q, s in zip(positions, tangents)])
        self.last_sensitivity_timings = dict(backend='cpu', transfer_ms=transfer_ms,
            replay_ms=1000*(time.perf_counter()-replay_started),
            predictor_graph_enabled=r.graph is not None)
        return result

    def final_points(self):
        r = self.runtime
        return r.state.particle_q.numpy()[r.particle_start:r.particle_start+r.particle_count].copy()

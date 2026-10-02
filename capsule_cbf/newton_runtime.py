"""Shared full-cloth Newton transition for live stepping and isolated rollouts.

Only immutable model data is shared. Every runtime owns states, contacts,
collision pipeline, Style3D workspaces, boundary arrays and CUDA graphs.
"""
import numpy as np
import time
import newton
import warp as wp


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
                 count: wp.array[int], center: wp.vec3, half_length: float,
                 radius: float, particle_radius: float, margin: float,
                 step: int, out: wp.array2d[float]):
    i = wp.tid()
    if i < count[0]:
        f, w = faces[i], weights[i]
        p = w[0]*q[triangles[f, 0]]+w[1]*q[triangles[f, 1]]+w[2]*q[triangles[f, 2]]
        a = wp.vec3(center[0], center[1], wp.clamp(p[2], center[2]-half_length,
                                                center[2]+half_length))
        out[step, i] = wp.length(p-a)-radius-particle_radius-margin


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
                      particle_radius: float, margin: float, step: int,
                      out: wp.array2d[float]):
    i = wp.tid()
    p = q[start+i]
    a = wp.vec3(center[0], center[1], wp.clamp(p[2], center[2]-half_length,
                                            center[2]+half_length))
    out[step, i] = wp.length(p-a)-radius-particle_radius-margin


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
            self.hit_count = wp.zeros(1, dtype=int, device=self.device)
            self.hit_observations = wp.zeros((substeps, capacity), dtype=float, device=self.device)
        self.graph = None
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
                          self.particle_radius, cfg.margin, step, self.observations],
                  device=self.device)
        if cfg.strain_enabled:
            wp.launch(observe_triangles, self.model.tri_count,
                      inputs=[state.particle_q, self.model.tri_indices, self.model.tri_poses,
                              self.particle_count, cfg.weft_strain_limit, cfg.warp_strain_limit,
                              cfg.shear_limit, step, self.observations], device=self.device)
        wp.launch(observe_hits, len(self.hit_faces),
                  inputs=[state.particle_q, self.model.tri_indices, self.hit_faces,
                          self.hit_weights, self.hit_count, wp.vec3(*c.center),
                          c.half_length, c.radius, self.particle_radius, cfg.margin,
                          step, self.hit_observations], device=self.device)

    def _advance(self):
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
        self.runtime = ClothRuntime(model, patch_indices, corner_ids, offsets,
                                    substeps, dt, iterations, linear_iterations,
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

    def begin(self, live_state, starts):
        self.snapshot.assign(live_state)
        self.starts = np.asarray(starts, dtype=np.float32).copy()
        self.calls = 0

    def __call__(self, followers, leaders):
        if self.starts is None:
            raise RuntimeError('Call begin before predicting')
        r = self.runtime
        r.state.assign(self.snapshot)
        r.scratch.assign(self.snapshot)
        r.advance(self.starts, np.vstack((followers, leaders)))
        self.calls += 1
        observations = (r.observations.numpy().astype(float) if self.selected_hit_count is None
                        else r.hit_observations.numpy()[:, :self.selected_hit_count].astype(float))
        # Isaac's main thread also owns its UI. Pump Kit between expensive
        # rollouts without advancing the live Newton state or changing the
        # leader command held fixed for this control frame.
        now = time.perf_counter()
        if self.progress_callback is not None and now-self.last_progress >= .05:
            self.progress_callback()
            self.last_progress = time.perf_counter()
        return observations

    def final_points(self):
        r = self.runtime
        return r.state.particle_q.numpy()[r.particle_start:r.particle_start+r.particle_count].copy()

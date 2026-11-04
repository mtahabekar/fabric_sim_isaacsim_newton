"""Newton Style3D cloth stepping with prescribed (velocity-commanded) gripper patches.

Used for the live cloth in Isaac Sim (fabric_newton_isaacsim.py) and for the
headless plant (scenario.py). Gripper particles are inactive in Newton's
solve; each substep places them at start + elapsed * command + offset, so the
patches follow their commands exactly. One frame of `substeps` steps can be
captured as a CUDA graph.
"""
import numpy as np
import newton
import warp as wp


@wp.kernel
def move_boundary(indices: wp.array[int], corner_ids: wp.array[int],
                  offsets: wp.array[wp.vec3], starts: wp.array[wp.vec3],
                  commands: wp.array[wp.vec3], elapsed: float,
                  q: wp.array[wp.vec3], qd: wp.array[wp.vec3]):
    i = wp.tid()
    c, j = corner_ids[i], indices[i]
    q[j] = starts[c]+elapsed*commands[c]+offsets[i]
    qd[j] = commands[c]


class ClothRuntime:
    """Owns the states, contacts, collision pipeline and CUDA graph of one cloth model."""

    def __init__(self, model, patch_indices, corner_ids, offsets, substeps, dt,
                 iterations, linear_iterations, *, solver=None, capture=True):
        if substeps < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive substeps and timestep required')
        self.model, self.device = model, model.device
        self.substeps, self.dt = substeps, dt
        # Newton 1.2.x precomputes the projective-dynamics matrix in __init__.
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
            a, b = b, a
        # Canonical output buffer, also for odd substep counts inside graphs.
        if a is not self.state:
            self.state.assign(a)

    def advance(self, starts, commands):
        """Advance one frame: corner c moves from starts[c] at velocity commands[c]."""
        starts, commands = np.asarray(starts), np.asarray(commands)
        if (starts.shape != (4, 3) or commands.shape != (4, 3)
                or not np.isfinite(starts).all() or not np.isfinite(commands).all()):
            raise ValueError('Invalid boundary state or commands')
        self.starts.assign(starts.astype(np.float32))
        self.commands.assign(commands.astype(np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._advance()

"""Style3D checkpoints in reusable device buffers, compatible with CUDA graphs.

Primal equations are unchanged. Host copies happen only when a tangent is
requested, in a fixed number of bulk transfers rather than per PCG iteration.
"""
import numpy as np
import warp as wp
from newton.solvers import SolverStyle3D
from newton._src.solvers.style3d.linear_solver import PcgSolver


@wp.kernel
def flag_contact(diagonal: wp.array[wp.mat33], flag: wp.array[int]):
    i = wp.tid()
    for a in range(3):
        for b in range(3):
            if diagonal[i][a, b] != 0.0:
                wp.atomic_max(flag, 0, 1)


@wp.kernel
def flag_body_contact(count: wp.array[int], flag: wp.array[int]):
    if count[0] > 0:
        flag[0] = 1


class RecordedPcg(PcgSolver):
    def __init__(self, owner):
        super().__init__(owner.model.particle_count, owner.device)
        self.owner = owner

    def solve(self, non_diag, diag, x0, b, inv_m, x1, iterations, additional_multiplier=None):
        o = self.owner
        self.record_index = o.substep_index*o.nonlinear_iterations+o.iteration_index
        if o.collision is not None:
            wp.launch(flag_contact, o.model.particle_count,
                      inputs=[o.collision.contact_hessian_diagonal(), o.contact_flag], device=o.device)
        wp.copy(o.q_tape[self.record_index], o.input_state.particle_q)
        # Contact-free A and preconditioner are constant throughout the frame.
        wp.copy(o.diag_tape, diag)
        wp.copy(o.inv_tape, inv_m)
        super().solve(non_diag, diag, x0, b, inv_m, x1, iterations, additional_multiplier)
        wp.copy(o.rz_tape[self.record_index], self.rTz, count=o.recorded_iterations)
        wp.copy(o.pap_tape[self.record_index], self.pTAp, count=o.recorded_iterations)
        o.iteration_index += 1

    def step7_update_x_r(self, x, iter):
        for j, source in enumerate((self.r, self.z, self.p, self.Ap)):
            wp.copy(self.owner.cg_tape[self.record_index, iter, j], source)
        super().step7_update_x_r(x, iter)


class SensitivityStyle3D(SolverStyle3D):
    def __init__(self, *args, substeps, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_solver = RecordedPcg(self)
        self.substeps = substeps
        self.recorded_iterations = min(self.linear_iterations, self.linear_solver.maxIter)
        records = substeps*self.nonlinear_iterations
        n = self.model.particle_count
        self.q_tape = wp.empty((records, n), dtype=wp.vec3, device=self.device)
        self.cg_tape = wp.empty((records, self.recorded_iterations, 4, n), dtype=wp.vec3, device=self.device)
        self.rz_tape = wp.empty((records, self.recorded_iterations), dtype=float, device=self.device)
        self.pap_tape = wp.empty_like(self.rz_tape)
        self.diag_tape = wp.empty(n, dtype=float, device=self.device)
        self.inv_tape = wp.empty(n, dtype=wp.mat33, device=self.device)
        self.position_tape = wp.empty((substeps, n), dtype=wp.vec3, device=self.device)
        self.contact_flag = wp.zeros(1, dtype=int, device=self.device)
        self.reset_trajectory()

    def invalidate_host(self):
        self._host_records = None
        self._host_positions = None

    def reset_trajectory(self):
        self.substep_index = 0
        self.iteration_index = 0
        self.contact_flag.zero_()
        self.invalidate_host()

    @property
    def contact_active(self):
        return bool(self.contact_flag.numpy()[0])

    @property
    def positions(self):
        if self._host_positions is None:
            self._host_positions = self.position_tape.numpy().astype(float)
        return self._host_positions

    @property
    def linear_records(self):
        if self._host_records is None:
            q, cg = self.q_tape.numpy(), self.cg_tape.numpy()
            rz, pap = self.rz_tape.numpy(), self.pap_tape.numpy()
            diag, inv = self.diag_tape.numpy(), self.inv_tape.numpy()
            self._host_records = []
            for s in range(self.substeps):
                records = []
                for i in range(self.nonlinear_iterations):
                    j = s*self.nonlinear_iterations+i
                    records.append(dict(q=q[j], diag=diag, inv_m=inv, initial=i == 0,
                                        rz=rz[j].astype(float), pap=pap[j].astype(float),
                                        steps=[tuple(cg[j, k, a] for a in range(4))
                                               for k in range(self.recorded_iterations)]))
                self._host_records.append(records)
        return self._host_records

    def step(self, state_in, state_out, control, contacts, dt):
        self.input_state = state_in
        self.iteration_index = 0
        super().step(state_in, state_out, control, contacts, dt)
        wp.copy(self.position_tape[self.substep_index], state_out.particle_q)
        wp.launch(flag_body_contact, 1, inputs=[contacts.soft_contact_count, self.contact_flag], device=self.device)
        self.substep_index += 1

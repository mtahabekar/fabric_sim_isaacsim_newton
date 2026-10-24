"""Six forward derivatives alongside Style3D, entirely on the model's device.

No iteration tape, NumPy replay, dense Jacobian, or perturbed primal trajectories.
Contact derivatives are deliberately unsupported; the predictor rejects them.
"""
import numpy as np
import warp as wp
from newton.solvers import SolverStyle3D
from newton._src.solvers.style3d.linear_solver import PcgSolver, SparseMatrixELL
from .sensitivity_solver import flag_contact, flag_body_contact
from .newton_runtime import boundary_geometry


@wp.func
def multiply(a: SparseMatrixELL, diagonal: wp.array[float],
             x: wp.array2d[wp.vec3], d: int, i: int):
    value = diagonal[i]*x[d, i]
    for k in range(a.num_nz[i]):
        entry = a.nz_ell[k, i]
        value += entry.value*x[d, entry.column_index]
    return value


@wp.kernel
def begin_substep(boundary: wp.array2d[wp.vec3], flags: wp.array[int], dt: float,
                  elapsed: float, sq: wp.array2d[wp.vec3], sv: wp.array2d[wp.vec3],
                  previous: wp.array2d[wp.vec3], inertia: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    if (flags[i] & 1) == 0:
        sq[d, i] = elapsed*boundary[d, i]
        sv[d, i] = boundary[d, i]
    previous[d, i] = sq[d, i]
    inertia[d, i] = sq[d, i]
    if (flags[i] & 1) != 0:
        inertia[d, i] += dt*sv[d, i]


@wp.kernel
def init_linear(mass: wp.array[float], flags: wp.array[int], dt: float, first: int,
                sq: wp.array2d[wp.vec3], sv: wp.array2d[wp.vec3], inertia: wp.array2d[wp.vec3],
                dr: wp.array2d[wp.vec3], dx: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    dr[d, i] = (mass[i]/(dt*dt))*(inertia[d, i]-sq[d, i])
    dx[d, i] = wp.vec3(0.0)
    if first != 0 and (flags[i] & 1) != 0:
        dx[d, i] = dt*sv[d, i]


@wp.kernel
def stretch_jvp(q: wp.array[wp.vec3], sq: wp.array2d[wp.vec3], faces: wp.array2d[int],
                rest: wp.array[wp.mat22], areas: wp.array[float], stiffness: wp.array[wp.vec3],
                dr: wp.array2d[wp.vec3], invalid: wp.array[int]):
    d, t = wp.tid()
    i0, i1, i2 = faces[t, 0], faces[t, 1], faces[t, 2]
    dm = rest[t]
    a = wp.vec3(-dm[0, 0]-dm[1, 0], dm[0, 0], dm[1, 0])
    b = wp.vec3(-dm[0, 1]-dm[1, 1], dm[0, 1], dm[1, 1])
    # Match upstream edge-based primal deformation arithmetic.
    fu = (q[i1]-q[i0])*a[1]+(q[i2]-q[i0])*a[2]
    fv = (q[i1]-q[i0])*b[1]+(q[i2]-q[i0])*b[2]
    dfu = (sq[d, i1]-sq[d, i0])*a[1]+(sq[d, i2]-sq[d, i0])*a[2]
    dfv = (sq[d, i1]-sq[d, i0])*b[1]+(sq[d, i2]-sq[d, i0])*b[2]
    lu, lv = wp.length(fu), wp.length(fv)
    if lu <= 1.e-6 or lv <= 1.e-6:
        wp.atomic_max(invalid, 0, 1)
        return
    u, v = fu/lu, fv/lv
    dlu, dlv = wp.dot(u, dfu), wp.dot(v, dfv)
    du, dv = (dfu-u*dlu)/lu, (dfv-v*dlv)/lv
    cosine = wp.dot(u, v)
    dc = wp.dot(du, v)+wp.dot(u, dv)
    for j in range(3):
        force = -areas[t]*(stiffness[t][0]*a[j]*(dlu*u+(lu-1.0)*du)
                            +stiffness[t][1]*b[j]*(dlv*v+(lv-1.0)*dv)
                            +stiffness[t][2]*(dc*(b[j]*u+a[j]*v)
                                             +cosine*(b[j]*du+a[j]*dv)))
        wp.atomic_add(dr, d, faces[t, j], force)


@wp.kernel
def bend_jvp(sq: wp.array2d[wp.vec3], edges: wp.array2d[int], cot: wp.array[wp.vec4],
             area: wp.array[float], properties: wp.array2d[float], dr: wp.array2d[wp.vec3]):
    d, e = wp.tid()
    if edges[e, 0] < 0 or edges[e, 1] < 0:
        return
    c = cot[e]
    w = wp.vec4(-c[0]-c[2], -c[1]-c[3], c[2]+c[3], c[0]+c[1])*(properties[e, 0]/area[e])
    for i in range(4):
        value = wp.vec3(0.0)
        for j in range(4):
            value -= w[i]*w[j]*sq[d, edges[e, j]]
        wp.atomic_add(dr, d, edges[e, i], value)


@wp.kernel
def initial_residual(a: SparseMatrixELL, diagonal: wp.array[float],
                     dx: wp.array2d[wp.vec3], dr: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    dr[d, i] -= multiply(a, diagonal, dx, d, i)


@wp.kernel
def precondition(inv: wp.array[wp.mat33], r: wp.array[wp.vec3], z: wp.array[wp.vec3],
                 dr: wp.array2d[wp.vec3], dz: wp.array2d[wp.vec3], terms: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    dz[d, i] = inv[i]*dr[d, i]
    # Store scalar partials as vec3 for upstream's native reduction helper.
    terms[d, i] = wp.vec3(wp.dot(z[i], dr[d, i])+wp.dot(r[i], dz[d, i]), 0.0, 0.0)


@wp.kernel
def direction_update(k: int, rz: wp.array[float], dnum: wp.array[float], old_dnum: wp.array[float],
                     old_p: wp.array[wp.vec3], dz: wp.array2d[wp.vec3], dp: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    beta, dbeta = float(0.0), float(0.0)
    if k > 0:
        num, denom = rz[k], rz[k-1]
        if wp.abs(denom) > 1.e-30 and not wp.isnan(denom) and not wp.isnan(num):
            beta = num/denom
            dbeta = (dnum[d]-beta*old_dnum[d])/denom
    dp[d, i] = dz[d, i]+beta*dp[d, i]+dbeta*old_p[i]


@wp.kernel
def product(a: SparseMatrixELL, diagonal: wp.array[float], p: wp.array[wp.vec3], ap: wp.array[wp.vec3],
            dp: wp.array2d[wp.vec3], dap: wp.array2d[wp.vec3], terms: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    dap[d, i] = multiply(a, diagonal, dp, d, i)
    terms[d, i] = wp.vec3(wp.dot(ap[i], dp[d, i])+wp.dot(p[i], dap[d, i]), 0.0, 0.0)


@wp.kernel
def cg_update(k: int, rz: wp.array[float], pap: wp.array[float], dnum: wp.array[float], dden: wp.array[float],
              p: wp.array[wp.vec3], ap: wp.array[wp.vec3], dp: wp.array2d[wp.vec3], dap: wp.array2d[wp.vec3],
              dx: wp.array2d[wp.vec3], dr: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    num, denom = rz[k], pap[k]
    alpha, dalpha = float(0.0), float(0.0)
    if wp.abs(denom) > 1.e-30 and not wp.isnan(denom) and not wp.isnan(num):
        alpha = num/denom
        dalpha = (dnum[d]-alpha*dden[d])/denom
    dx[d, i] += alpha*dp[d, i]+dalpha*p[i]
    dr[d, i] -= alpha*dap[d, i]+dalpha*ap[i]


@wp.kernel
def add_increment(sq: wp.array2d[wp.vec3], dx: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    sq[d, i] += dx[d, i]


@wp.kernel
def finish_substep(sq: wp.array2d[wp.vec3], previous: wp.array2d[wp.vec3], boundary: wp.array2d[wp.vec3],
                   flags: wp.array[int], dt: float, sv: wp.array2d[wp.vec3]):
    d, i = wp.tid()
    sv[d, i] = 0.998*(sq[d, i]-previous[d, i])/dt
    if (flags[i] & 1) == 0:
        sv[d, i] = boundary[d, i]


@wp.kernel
def particle_observation(q: wp.array[wp.vec3], sq: wp.array2d[wp.vec3], start: int,
                         center: wp.vec3, h: float, r: float, mode: int, step: int,
                         out: wp.array3d[float]):
    d, i = wp.tid()
    distance, normal = boundary_geometry(q[start+i], center, h, r, mode)
    out[step, i, d] = wp.dot(normal, sq[d, start+i])


@wp.kernel
def hit_observation(q: wp.array[wp.vec3], sq: wp.array2d[wp.vec3], triangles: wp.array2d[int],
                    faces: wp.array[int], weights: wp.array[wp.vec3], offsets: wp.array[wp.vec3], count: wp.array[int],
                    center: wp.vec3, h: float, r: float, mode: int, step: int, out: wp.array3d[float]):
    d, i = wp.tid()
    if i < count[0]:
        f, w = faces[i], weights[i]
        a, b, c = triangles[f, 0], triangles[f, 1], triangles[f, 2]
        point = offsets[i]+w[0]*q[a]+w[1]*q[b]+w[2]*q[c]
        tangent = w[0]*sq[d, a]+w[1]*sq[d, b]+w[2]*sq[d, c]
        distance, normal = boundary_geometry(point, center, h, r, mode)
        out[step, i, d] = wp.dot(normal, tangent)


@wp.kernel
def strain_observation(q: wp.array[wp.vec3], sq: wp.array2d[wp.vec3], faces: wp.array2d[int],
                       rest: wp.array[wp.mat22], offset: int, step: int,
                       out: wp.array3d[float], invalid: wp.array[int]):
    d, t = wp.tid()
    a, b, c = faces[t, 0], faces[t, 1], faces[t, 2]
    dm = rest[t]
    fu, fv = ((q[b]-q[a])*dm[0, 0]+(q[c]-q[a])*dm[1, 0],
              (q[b]-q[a])*dm[0, 1]+(q[c]-q[a])*dm[1, 1])
    dfu, dfv = ((sq[d, b]-sq[d, a])*dm[0, 0]+(sq[d, c]-sq[d, a])*dm[1, 0],
                (sq[d, b]-sq[d, a])*dm[0, 1]+(sq[d, c]-sq[d, a])*dm[1, 1])
    lu, lv = wp.length(fu), wp.length(fv)
    if lu <= 1.e-6 or lv <= 1.e-6:
        wp.atomic_max(invalid, 0, 1)
        return
    u, v = fu/lu, fv/lv
    dlu, dlv = wp.dot(u, dfu), wp.dot(v, dfv)
    dc = wp.dot((dfu-u*dlu)/lu, v)+wp.dot(u, (dfv-v*dlv)/lv)
    j = offset+4*t
    out[step, j, d] = -dlu
    out[step, j+1, d] = -dlv
    out[step, j+2, d] = -dc
    out[step, j+3, d] = dc


class TangentPcg(PcgSolver):
    def __init__(self, owner):
        super().__init__(owner.model.particle_count, owner.device)
        self.owner = owner
        self.old_p = wp.empty(self.dim, dtype=wp.vec3, device=self.device)
        self.ones = wp.array(np.tile([1., 0., 0.], (self.dim, 1)), dtype=wp.vec3, device=self.device)

    def reduce(self, output):
        # Reuse the same native device reduction as primal PCG, avoiding one
        # contended global atomic per particle per direction.
        from newton._src.solvers.style3d.linear_solver import array_inner
        for d in range(6):
            array_inner(self.owner.terms[d], self.ones, output.ptr+d*output.strides[0])

    def solve(self, non_diag, diag, x0, b, inv_m, x1, iterations, additional_multiplier=None):
        o, m = self.owner, self.owner.model
        if o.collision is not None:
            o.launch(flag_contact, m.particle_count, [o.collision.contact_hessian_diagonal(), o.contact_flag])
        o.dp.zero_()
        o.old_dnum.zero_()
        o.launch(init_linear, o.dim, [m.particle_mass, m.particle_flags, o.dt, int(x0 is not None),
                                     o.sq, o.sv, o.inertia, o.dr, o.dx_tangent])
        o.launch(stretch_jvp, (6, m.tri_count), [o.input_state.particle_q, o.sq, m.tri_indices,
                    m.tri_poses, m.tri_areas, m.style3d.tri_aniso_ke, o.dr, o.invalid_flag])
        o.launch(bend_jvp, (6, m.edge_count), [o.sq, m.edge_indices, m.style3d.edge_bending_cot,
                    m.style3d.edge_rest_area, m.edge_bending_properties, o.dr])
        o.launch(initial_residual, o.dim, [non_diag, diag, o.dx_tangent, o.dr])
        super().solve(non_diag, diag, x0, b, inv_m, x1, iterations, additional_multiplier)
        o.launch(add_increment, o.dim, [o.sq, o.dx_tangent])

    def step2_update_z(self, inv_M):
        super().step2_update_z(inv_M)
        o = self.owner
        o.launch(precondition, o.dim, [inv_M, self.r, self.z, o.dr, o.dz, o.terms])
        self.reduce(o.dnum)

    def step4_update_p(self, iter):
        o = self.owner
        if iter == 0:
            self.old_p.zero_()
        else:
            wp.copy(self.old_p, self.p)
        super().step4_update_p(iter)
        o.launch(direction_update, o.dim, [iter, self.rTz, o.dnum, o.old_dnum, self.old_p, o.dz, o.dp])
        wp.copy(o.old_dnum, o.dnum)

    def step5_update_Ap(self, A_non_diag, A_diag, additional_Ap=None):
        super().step5_update_Ap(A_non_diag, A_diag, additional_Ap)
        o = self.owner
        o.launch(product, o.dim, [A_non_diag, A_diag, self.p, self.Ap, o.dp, o.dap, o.terms])
        self.reduce(o.dden)

    def step7_update_x_r(self, x, iter):
        o = self.owner
        o.launch(cg_update, o.dim, [iter, self.rTz, self.pTAp, o.dnum, o.dden,
                                   self.p, self.Ap, o.dp, o.dap, o.dx_tangent, o.dr])
        super().step7_update_x_r(x, iter)


class TangentStyle3D(SolverStyle3D):
    """Finite-iteration derivative on CPU/CUDA Warp, with O(6*N) workspace."""
    def __init__(self, model, patch_indices, corner_ids, **kwargs):
        super().__init__(model, **kwargs)
        self.dim = (6, model.particle_count)
        for name in ('sq', 'sv', 'previous', 'inertia', 'dr', 'dz', 'dp', 'dap', 'dx_tangent', 'terms'):
            setattr(self, name, wp.zeros(self.dim, dtype=wp.vec3, device=self.device))
        for name in ('dnum', 'old_dnum', 'dden'):
            setattr(self, name, wp.zeros(6, dtype=float, device=self.device))
        self.contact_flag = wp.zeros(1, dtype=int, device=self.device)
        self.invalid_flag = wp.zeros(1, dtype=int, device=self.device)
        boundary = np.zeros((6, model.particle_count, 3), dtype=np.float32)
        for index, corner in zip(patch_indices, corner_ids):
            if corner < 2:
                boundary[3*corner:3*corner+3, index] = np.eye(3)
        self.boundary = wp.array(boundary, dtype=wp.vec3, device=self.device)
        self.linear_solver = TangentPcg(self)
        self.reset_trajectory()

    def launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim, inputs=inputs, device=self.device)

    def invalidate_host(self):
        pass  # no checkpoint or host derivative cache

    def reset_trajectory(self):
        self.sq.zero_()
        self.sv.zero_()
        self.contact_flag.zero_()
        self.invalid_flag.zero_()
        self.substep_index = 0

    @property
    def contact_active(self):
        return bool(self.contact_flag.numpy()[0])

    def step(self, state_in, state_out, control, contacts, dt):
        self.input_state, self.dt = state_in, dt
        self.launch(begin_substep, self.dim, [self.boundary, self.model.particle_flags,
                    dt, (self.substep_index+1)*dt, self.sq, self.sv, self.previous, self.inertia])
        super().step(state_in, state_out, control, contacts, dt)
        self.launch(finish_substep, self.dim, [self.sq, self.previous, self.boundary,
                    self.model.particle_flags, dt, self.sv])
        self.launch(flag_body_contact, 1, [contacts.soft_contact_count, self.contact_flag])
        self.substep_index += 1

    def configure_observations(self, runtime):
        self.jacobian = wp.zeros((*runtime.observations.shape, 6), dtype=float, device=self.device)
        self.hit_jacobian = wp.zeros((*runtime.hit_observations.shape, 6), dtype=float, device=self.device)

    def observe_tangent(self, runtime, state, step):
        r, c, m = runtime, runtime.capsule, self.model
        geometry = [wp.vec3(*c.center), c.half_length, c.radius, int(c.centered_top), step]
        self.launch(particle_observation, (6, r.particle_count),
                    [state.particle_q, self.sq, r.particle_start, *geometry, self.jacobian])
        self.launch(hit_observation, (6, len(r.hit_faces)),
                    [state.particle_q, self.sq, m.tri_indices, r.hit_faces, r.hit_weights,
                     r.hit_offsets, r.hit_count, *geometry, self.hit_jacobian])
        if r.settings.strain_enabled:
            self.launch(strain_observation, (6, m.tri_count),
                        [state.particle_q, self.sq, m.tri_indices, m.tri_poses,
                         r.particle_count, step, self.jacobian, self.invalid_flag])

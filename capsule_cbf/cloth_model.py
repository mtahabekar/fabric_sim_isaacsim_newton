"""Style3D-consistent cloth mechanics and a linearized short-horizon model.

Pure NumPy/SciPy at runtime; Newton is touched only by ``from_newton``.
Forces reproduce Newton's eval_stretch_kernel and eval_bend_kernel. The
horizon model is linearly implicit Euler (Baraff and Witkin, 1998) with one
frozen, PSD-projected tangent stiffness per control update. See
SENSITIVITY_CBF_REPORT.md for the derivation.
"""
import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse import coo_matrix, csr_matrix, kron, identity
from scipy.sparse.csgraph import reverse_cuthill_mckee
from .dynamics_sensitivity import stretch_force_blocks


class ClothMechanics:
    """Internal forces f(q) and stiffness K(q) = -df/dq of one cloth mesh."""

    def __init__(self, triangles, inv_dm, areas, aniso_ke, bend_edges, bend_weights,
                 mass, pinned, gravity):
        self.triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
        self.inv_dm = np.asarray(inv_dm, dtype=float).reshape(-1, 2, 2)
        self.areas = np.asarray(areas, dtype=float).ravel()
        self.aniso_ke = np.asarray(aniso_ke, dtype=float).reshape(-1, 3)
        self.mass = np.asarray(mass, dtype=float).ravel()
        self.pinned = np.asarray(pinned, dtype=bool).ravel()
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        n, t = len(self.mass), len(self.triangles)
        if (len(self.inv_dm) != t or len(self.areas) != t or len(self.aniso_ke) != t
                or len(self.pinned) != n or self.triangles.min(initial=0) < 0
                or self.triangles.max(initial=0) >= n):
            raise ValueError('Inconsistent cloth mesh arrays')
        self.count = n
        # Fu = sum_i a_i x_i, Fv = sum_i b_i x_i, exactly as the Warp kernel.
        self.coeff = np.concatenate((-self.inv_dm.sum(axis=1)[:, None, :], self.inv_dm), axis=1)
        corners = self.triangles.ravel()
        self.scatter = csr_matrix((np.ones(3*t), (corners, np.arange(3*t))), shape=(n, 3*t))
        edges = np.asarray(bend_edges, dtype=int).reshape(-1, 4)
        weights = np.asarray(bend_weights, dtype=float).reshape(-1, 4)
        blocks = weights[:, :, None]*weights[:, None, :]
        rows = np.broadcast_to(edges[:, :, None], blocks.shape)
        cols = np.broadcast_to(edges[:, None, :], blocks.shape)
        # Quadratic isometric bending: f_bend = -Q q with constant PSD Q.
        self.bend = coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())), shape=(n, n)).tocsr()
        self.bend3 = kron(self.bend, identity(3), format='csr')
        dofs = 3*self.triangles[:, :, None]+np.arange(3)
        self.k_rows = np.broadcast_to(dofs.reshape(t, 9, 1), (t, 9, 9)).ravel()
        self.k_cols = np.broadcast_to(dofs.reshape(t, 1, 9), (t, 9, 9)).ravel()

    @classmethod
    def from_newton(cls, model, start=0, count=None):
        """Copy one contiguous cloth's arrays from a finalized Newton model."""
        count = model.particle_count-start if count is None else count
        triangles = model.tri_indices.numpy()-start
        keep = np.all((triangles >= 0) & (triangles < count), axis=1)
        edges = model.edge_indices.numpy()
        valid = np.all(edges >= 0, axis=1)
        cot = model.style3d.edge_bending_cot.numpy()[valid].astype(float)
        weights = np.column_stack((-cot[:, 0]-cot[:, 2], -cot[:, 1]-cot[:, 3],
                                   cot[:, 2]+cot[:, 3], cot[:, 0]+cot[:, 1]))
        weights *= (model.edge_bending_properties.numpy()[valid, 0]
                    /model.style3d.edge_rest_area.numpy()[valid])[:, None]
        flags = model.particle_flags.numpy()[start:start+count]
        return cls(triangles[keep], model.tri_poses.numpy()[keep], model.tri_areas.numpy()[keep],
                   model.style3d.tri_aniso_ke.numpy()[keep], edges[valid]-start, weights,
                   model.particle_mass.numpy()[start:start+count], (flags & 1) == 0,
                   model.gravity.numpy().reshape(-1, 3)[0])

    def _frames(self, q, subset=slice(None)):
        p = np.asarray(q, dtype=float)[self.triangles[subset]]
        fu = np.einsum('tix,ti->tx', p, self.coeff[subset, :, 0])
        fv = np.einsum('tix,ti->tx', p, self.coeff[subset, :, 1])
        lu, lv = np.linalg.norm(fu, axis=1), np.linalg.norm(fv, axis=1)
        u = np.where(lu[:, None] > 1e-6, fu/np.maximum(lu, 1e-12)[:, None], 0.)
        v = np.where(lv[:, None] > 1e-6, fv/np.maximum(lv, 1e-12)[:, None], 0.)
        return u, v, lu, lv

    def corner_forces(self, q, subset=slice(None)):
        """Stretch force on each triangle corner, shape (T, 3, 3)."""
        u, v, lu, lv = self._frames(q, subset)
        k, a, b = self.aniso_ke[subset], self.coeff[subset, :, 0], self.coeff[subset, :, 1]
        cosine = np.einsum('tx,tx->t', u, v)
        su = (k[:, 0]*(lu-1.))[:, None]*a
        sv = (k[:, 1]*(lv-1.))[:, None]*b
        return -self.areas[subset, None, None]*(
            su[:, :, None]*u[:, None, :]+sv[:, :, None]*v[:, None, :]
            +(k[:, 2]*cosine)[:, None, None]*(b[:, :, None]*u[:, None, :]+a[:, :, None]*v[:, None, :]))

    def _incident(self, nodes):
        key = tuple(np.asarray(nodes, dtype=int).tolist())
        cache = self.__dict__.setdefault('_incident_cache', {})
        if key not in cache:
            member = np.zeros(self.count, dtype=bool)
            member[list(key)] = True
            tris = np.flatnonzero(member[self.triangles].any(axis=1))
            cache[key] = (tris, member[self.triangles[tris]], member)
        return cache[key]

    def resultant(self, q, nodes):
        """Sum of internal forces on a node set: the cloth's pull on a gripper."""
        tris, corner_in, member = self._incident(nodes)
        forces = self.corner_forces(q, tris)
        return forces[corner_in].sum(axis=0)-(self.bend[member]@np.asarray(q, dtype=float)).sum(axis=0)

    def resultant_jacobian(self, q, nodes):
        """Exact d(resultant)/dq, shape (3, N, 3)."""
        tris, corner_in, member = self._incident(nodes)
        blocks = stretch_force_blocks(np.asarray(q, dtype=float), self.triangles[tris], self.inv_dm[tris],
                                      self.areas[tris], self.aniso_ke[tris])  # (t, i, j, x, y)
        local = np.einsum('ti,tijxy->tjxy', corner_in.astype(float), blocks)
        out = np.zeros((self.count, 3, 3))
        np.add.at(out, self.triangles[tris], local)
        bend = np.asarray(self.bend[member].sum(axis=0)).ravel()
        out -= bend[:, None, None]*np.eye(3)[None]
        return out.transpose(1, 0, 2)

    def forces(self, q):
        """Total internal force per particle (stretch + bending), shape (N, 3)."""
        q = np.asarray(q, dtype=float)
        return self.scatter@self.corner_forces(q).reshape(-1, 3)-self.bend@q

    def force_jacobian_blocks(self, q):
        """Exact df/dx stretch blocks, shape (T, 9, 9), rows (vertex, xyz)."""
        blocks = stretch_force_blocks(np.asarray(q, dtype=float), self.triangles, self.inv_dm,
                                      self.areas, self.aniso_ke)
        return blocks.transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)

    def stiffness(self, q, project=True, blocks=None):
        """K = -df/dq (3N x 3N CSR). Projection makes each element PSD.

        The installed shear force is not an exact energy gradient, so element
        blocks are symmetrized before eigenvalue clamping (Teran et al. 2005).
        """
        k = -(self.force_jacobian_blocks(q) if blocks is None else blocks)
        if project:
            k = self.project(k)
        stretch = coo_matrix((k.ravel(), (self.k_rows, self.k_cols)), shape=(3*self.count,)*2)
        return (stretch.tocsr()+self.bend3).tocsr()

    @staticmethod
    def project(k):
        """Symmetrize and clamp negative eigenvalues of (T, 9, 9) blocks."""
        k = .5*(k+k.transpose(0, 2, 1))
        w, vectors = np.linalg.eigh(k)
        return np.einsum('tij,tj,tkj->tik', vectors, np.maximum(w, 0.), vectors, optimize=False)

    def jacobian_product(self, q, directions, blocks=None):
        """Matrix-free (df/dq) @ directions, directions shape (N, 3, C)."""
        d = np.asarray(directions, dtype=float)
        blocks = self.force_jacobian_blocks(q) if blocks is None else blocks
        local = d[self.triangles].reshape(len(self.triangles), 9, -1)
        corner = np.einsum('tij,tjc->tic', blocks, local, optimize=False)
        out = self.scatter@corner.reshape(3*len(self.triangles), -1)
        return out.reshape(d.shape)-(self.bend@d.reshape(self.count, -1)).reshape(d.shape)


class HorizonModel:
    """Linearly implicit Euler prediction with 12 corner-velocity sensitivities.

    For frozen K and step h, free DOFs obey
        (M + h^2 K_ff) v+ = M v + h [f_f(q_f, p(t+h)) + M g + b],  q_f+ = q_f + h v+,
    then v+ is scaled by the plant's per-step velocity damping. K is the
    PSD-projected stiffness: it only stabilizes the implicit solve (a
    Rosenbrock-W matrix). Sensitivities S = d(q, v)/d(corner velocity) are the
    derivative of this scheme, so they use the EXACT force Jacobian J = df/dq:
        (M + h^2 K_ff) S_v+ = M S_v + h (J_ff S_q + J_fp Pi(t+h)),  Pi(t) = t D.
    jacobian='trajectory' re-evaluates J at every predicted state (forward
    sensitivity analysis, exact for the scheme); 'frozen' keeps J(q0).
    jacobian='affine' (default) replaces f by f(q0) - K (q - q0) with the
    projected K for drift and sensitivities alike: a linear SPD system, so
    implicit Euler is unconditionally stable. The nonlinear modes diverge once
    compressed triangles (negative geometric stiffness) are integrated at a
    frame-sized step; see SENSITIVITY_CBF_REPORT.md, step 3.
    Columns 0:6 are follower velocities and 6:12 leader velocities.
    """

    def __init__(self, mechanics, pin_nodes, pin_corners, pin_offsets, frame_dt, substeps=1,
                 plant_dt=None, plant_damping=.998, jacobian='affine'):
        if jacobian not in ('affine', 'trajectory', 'frozen'):
            raise ValueError('jacobian must be affine, trajectory or frozen')
        self.jacobian_mode = jacobian
        self.m = mechanics
        self.pin_nodes = np.asarray(pin_nodes, dtype=int)
        self.pin_corners = np.asarray(pin_corners, dtype=int)
        self.pin_offsets = np.asarray(pin_offsets, dtype=float).reshape(-1, 3)
        if (len(self.pin_nodes) != len(self.pin_corners) or len(self.pin_nodes) != len(self.pin_offsets)
                or not np.array_equal(np.sort(self.pin_nodes), np.flatnonzero(mechanics.pinned))):
            raise ValueError('Pin nodes must be exactly the inactive mechanics nodes')
        if substeps < 1 or frame_dt <= 0:
            raise ValueError('Positive frame_dt and substeps required')
        self.frame_dt, self.substeps = frame_dt, substeps
        self.h = frame_dt/substeps
        plant_dt = self.h if plant_dt is None else plant_dt
        self.damping = plant_damping**(self.h/plant_dt)
        n = mechanics.count
        self.free = np.flatnonzero(~mechanics.pinned)
        self.free_dofs = (3*self.free[:, None]+np.arange(3)).ravel()
        self.pin_dofs = (3*self.pin_nodes[:, None]+np.arange(3)).ravel()
        # Pin displacement per unit time and unit corner velocity.
        self.pin_map = np.zeros((len(self.pin_dofs), 12))
        for k, corner in enumerate(self.pin_corners):
            self.pin_map[3*k:3*k+3, 3*corner:3*corner+3] = np.eye(3)
        self.mass_free = np.repeat(mechanics.mass[self.free], 3)
        self.count = n
        self.factor = None
        self._assembly_pattern()

    def _assembly_pattern(self):
        """Fixed-topology slots: each update is one bincount, no sparse build.

        K_ff is factored as an RCM-ordered banded Cholesky matrix; A = M + h^2 K
        is SPD because K is projected and free masses are positive.
        """
        m = self.m
        bend = m.bend3.tocoo()
        self.bend_data = bend.data
        rows = np.concatenate((m.k_rows, bend.row))
        cols = np.concatenate((m.k_cols, bend.col))
        nf, npd = len(self.free_dofs), len(self.pin_dofs)
        free_of = np.full(3*m.count, -1)
        free_of[self.free_dofs] = np.arange(nf)
        pin_of = np.full(3*m.count, -1)
        pin_of[self.pin_dofs] = np.arange(npd)
        fr, fc, pc = free_of[rows], free_of[cols], pin_of[cols]
        self.ff_mask, self.fp_mask = (fr >= 0) & (fc >= 0), (fr >= 0) & (pc >= 0)
        keys, self.ff_slot = np.unique(fr[self.ff_mask]*nf+fc[self.ff_mask], return_inverse=True)
        self.ff_rows, self.ff_cols = np.divmod(keys, nf)
        self.ff_indptr = np.searchsorted(self.ff_rows, np.arange(nf+1))
        keys, self.fp_slot = np.unique(fr[self.fp_mask]*npd+pc[self.fp_mask], return_inverse=True)
        fp_rows, self.fp_cols = np.divmod(keys, npd)
        self.fp_indptr = np.searchsorted(fp_rows, np.arange(nf+1))
        pattern = csr_matrix((np.ones(len(self.ff_cols)), self.ff_cols, self.ff_indptr), shape=(nf, nf))
        self.perm = reverse_cuthill_mckee(pattern, symmetric_mode=True)
        inverse = np.empty(nf, dtype=int)
        inverse[self.perm] = np.arange(nf)
        pi, pj = inverse[self.ff_rows], inverse[self.ff_cols]
        upper = pi <= pj
        self.band = int(np.max(pj-pi))
        self.band_upper = upper
        self.band_slot = (self.band+pi[upper]-pj[upper])*nf+pj[upper]

    def pins(self, corner_start, corner_velocity, t):
        return (np.asarray(corner_start)[self.pin_corners]+self.pin_offsets
                +t*np.asarray(corner_velocity)[self.pin_corners])

    def linearize(self, q):
        """Factor (M + h^2 K_ff) once; exact J only for non-affine modes."""
        q = np.asarray(q, dtype=float)
        blocks = self.m.force_jacobian_blocks(q)
        values = np.concatenate((self.m.project(-blocks).ravel(), self.bend_data))
        nf = len(self.free_dofs)
        ff = np.bincount(self.ff_slot, values[self.ff_mask], minlength=len(self.ff_cols))
        fp = np.bincount(self.fp_slot, values[self.fp_mask], minlength=len(self.fp_cols))
        self.k_ff = csr_matrix((ff, self.ff_cols, self.ff_indptr), shape=(nf, nf))
        self.k_fp = csr_matrix((fp, self.fp_cols, self.fp_indptr), shape=(nf, len(self.pin_dofs)))
        ab = np.bincount(self.band_slot, self.h**2*ff[self.band_upper],
                         minlength=(self.band+1)*nf).reshape(self.band+1, nf)
        ab[self.band] += self.mass_free[self.perm]
        self.factor = cholesky_banded(ab, lower=False)
        if self.jacobian_mode != 'affine':
            jac = -self.m.stiffness(q, project=False, blocks=blocks)
            rows = jac[self.free_dofs]
            self.j_ff = rows[:, self.free_dofs].tocsr()
            self.j_fp = rows[:, self.pin_dofs].tocsr()
        self.q_lin = q.copy()
        self.f_lin = self.m.forces(q)[self.free].ravel()

    def solve(self, rhs):
        out = np.empty_like(rhs)
        out[self.perm] = cho_solve_banded((self.factor, False), rhs[self.perm])
        return out

    def predict(self, q0, v0, corner_start, corner_velocity, frames, bias=None):
        """Return positions, velocities and sensitivities at frames 1..N.

        Shapes: q, v (N, nodes, 3); sq, sv (N, nodes, 3, 12). Pinned rows hold
        the exact prescribed motion and its sensitivity.
        """
        if self.factor is None:
            raise RuntimeError('Call linearize before predict')
        q = np.asarray(q0, dtype=float).copy()
        vf = np.asarray(v0, dtype=float)[self.free].ravel()
        qf = q[self.free].ravel()
        nf = len(self.free_dofs)
        sq, sv = np.zeros((nf, 12)), np.zeros((nf, 12))
        gravity = np.tile(self.m.gravity, len(self.free))*self.mass_free
        bias = np.zeros(nf) if bias is None else np.asarray(bias, dtype=float).reshape(-1)
        out_q, out_v = np.empty((frames, self.count, 3)), np.empty((frames, self.count, 3))
        out_sq, out_sv = np.zeros((frames, self.count, 3, 12)), np.zeros((frames, self.count, 3, 12))
        h = self.h
        full = np.zeros((self.count, 3, 12))
        affine = self.jacobian_mode == 'affine'
        qf_lin = self.q_lin[self.free].ravel()
        p_lin = self.q_lin[self.pin_nodes].ravel()
        for step in range(1, frames*self.substeps+1):
            t = step*h
            q[self.pin_nodes] = self.pins(corner_start, corner_velocity, t)
            q[self.free] = qf.reshape(-1, 3)
            pin_sens = t*self.pin_map
            if affine:
                force = (self.f_lin-self.k_ff@(qf-qf_lin)
                         -self.k_fp@(q[self.pin_nodes].ravel()-p_lin)+gravity+bias)
                product = -(self.k_ff@sq+self.k_fp@pin_sens)
            elif self.jacobian_mode == 'frozen':
                force = self.m.forces(q)[self.free].ravel()+gravity+bias
                product = self.j_ff@sq+self.j_fp@pin_sens
            else:
                force = self.m.forces(q)[self.free].ravel()+gravity+bias
                full[self.free] = sq.reshape(-1, 3, 12)
                full[self.pin_nodes] = pin_sens.reshape(-1, 3, 12)
                product = self.m.jacobian_product(q, full)[self.free].reshape(-1, 12)
            rhs = np.column_stack((self.mass_free*vf+h*force,
                                   self.mass_free[:, None]*sv+h*product))
            solution = self.solve(rhs)
            vf, sv = solution[:, 0], solution[:, 1:]
            qf = qf+h*vf
            sq = sq+h*sv
            vf, sv = self.damping*vf, self.damping*sv
            if step % self.substeps == 0:
                k = step//self.substeps-1
                q[self.free] = qf.reshape(-1, 3)
                out_q[k] = q
                out_v[k, self.free] = vf.reshape(-1, 3)
                out_v[k, self.pin_nodes] = np.asarray(corner_velocity)[self.pin_corners]
                out_sq[k, self.free] = sq.reshape(-1, 3, 12)
                out_sq[k, self.pin_nodes] = pin_sens.reshape(-1, 3, 12)
                out_sv[k, self.free] = sv.reshape(-1, 3, 12)
                out_sv[k, self.pin_nodes] = self.pin_map.reshape(-1, 3, 12)
        return dict(q=out_q, v=out_v, sq=out_sq, sv=out_sv)

    def one_step_response(self, residual):
        """Force bias b whose one-step response equals a position residual.

        For a constant force b over one implicit frame step D,
        dq = D^2 (M + D^2 K)^-1 b, hence b = (M + D^2 K) dq / D^2 on free DOFs.
        """
        r = np.asarray(residual, dtype=float)[self.free].ravel()
        d = self.frame_dt
        return (self.mass_free*r+d**2*(self.k_ff@r))/d**2

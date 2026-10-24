"""Contact-free Style3D derivatives evaluated along a nominal trajectory.

The solver tangent differentiates the actual finite PD/PCG iterations.
See DYNAMICS_SENSITIVITY.md for scope, equations and validation.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, kron, eye


def as_numpy(value):
    return value if isinstance(value, np.ndarray) else value.numpy()


class SensitivityUnavailable(RuntimeError):
    """The controller must use its finite-difference fallback."""


def element_matrix(indices, blocks, count):
    """Assemble (element, local vertex, local vertex, xyz, xyz) blocks."""
    dofs = 3*np.asarray(indices)[..., None]+np.arange(3)
    shape = blocks.shape
    rows = np.broadcast_to(dofs[:, :, None, :, None], shape)
    cols = np.broadcast_to(dofs[:, None, :, None, :], shape)
    return coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(3*count, 3*count)).tocsc()


def stretch_force_jacobian(q, faces, rest, areas, stiffness):
    """Exact derivative of the installed eval_stretch_kernel force formula.

    In particular its shear force uses NORMALIZED material directions. It is
    not replaced by a conventional unnormalized shear energy Hessian.
    """
    coeff = np.concatenate((-rest.sum(axis=1)[:, None, :], rest), axis=1)
    fu = np.einsum('tix,ti->tx', q[faces], coeff[:, :, 0])
    fv = np.einsum('tix,ti->tx', q[faces], coeff[:, :, 1])
    lu, lv = np.linalg.norm(fu, axis=1), np.linalg.norm(fv, axis=1)
    if np.any(lu <= 1e-6) or np.any(lv <= 1e-6):
        raise SensitivityUnavailable('degenerate material direction')
    u, v = fu/lu[:, None], fv/lv[:, None]
    eye = np.eye(3)[None]
    uu, vv = u[:, :, None]*u[:, None, :], v[:, :, None]*v[:, None, :]
    du, dv = (eye-uu)/lu[:, None, None], (eye-vv)/lv[:, None, None]
    tu = uu+(lu-1)[:, None, None]*du
    tv = vv+(lv-1)[:, None, None]*dv
    cosine = np.einsum('ti,ti->t', u, v)
    blocks = np.empty((len(faces), 3, 3, 3, 3))
    for i in range(3):
        ai, bi = coeff[:, i, 0], coeff[:, i, 1]
        w = bi[:, None]*u+ai[:, None]*v
        for j in range(3):
            aj, bj = coeff[:, j, 0], coeff[:, j, 1]
            dc = (aj[:, None]*np.einsum('txy,ty->tx', du, v)
                  +bj[:, None]*np.einsum('txy,ty->tx', dv, u))
            shear = w[:, :, None]*dc[:, None, :]+cosine[:, None, None]*(
                (bi*aj)[:, None, None]*du+(ai*bj)[:, None, None]*dv)
            blocks[:, i, j] = -areas[:, None, None]*(
                (stiffness[:, 0]*ai*aj)[:, None, None]*tu
                +(stiffness[:, 1]*bi*bj)[:, None, None]*tv
                +stiffness[:, 2, None, None]*shear)
    return element_matrix(faces, blocks, len(q))


class ClothDynamicsSensitivity:
    def __init__(self, model, patch_indices, corner_ids):
        self.count = model.particle_count
        self.faces = model.tri_indices.numpy()
        self.rest = model.tri_poses.numpy().astype(float)
        self.areas = model.tri_areas.numpy().astype(float)
        self.stiffness = model.style3d.tri_aniso_ke.numpy().astype(float)
        self.mass = np.repeat(model.particle_mass.numpy().astype(float), 3)
        active = (model.particle_flags.numpy() & 1) != 0
        self.fixed = np.flatnonzero(np.repeat(~active, 3))
        self.boundary = np.zeros((self.count, 3, 6))
        for index, corner in zip(patch_indices, corner_ids):
            if active[index]:
                raise ValueError('Sensitivity requires inactive prescribed pin particles')
            if corner < 2:
                self.boundary[index, :, 3*corner:3*corner+3] = np.eye(3)
        self.boundary = self.boundary.reshape(-1, 6)
        edges = model.edge_indices.numpy()
        valid = (edges[:, :2] >= 0).all(axis=1)
        edges = edges[valid]
        cot = model.style3d.edge_bending_cot.numpy()[valid].astype(float)
        weights = np.column_stack((-cot[:, 0]-cot[:, 2], -cot[:, 1]-cot[:, 3],
                                    cot[:, 2]+cot[:, 3], cot[:, 0]+cot[:, 1]))
        weights *= (model.edge_bending_properties.numpy()[valid, 0]
                    /model.style3d.edge_rest_area.numpy()[valid])[:, None]
        blocks = -np.einsum('ti,tj,ab->tijab', weights, weights, np.eye(3))
        self.bend = element_matrix(edges, blocks, self.count)

    def solver_trajectory(self, solver, dt, progress=None):
        """Forward differentiation of the recorded finite PD and PCG sequence.

        The constant contact-free PD matrix has zero input derivative. PCG
        coefficients do NOT: differentiating alpha and beta is essential when
        the solve is truncated. Primal values come from upstream checkpoints.
        """
        nz = solver.pd_non_diags.nz_ell.numpy()
        counts = solver.pd_non_diags.num_nz.numpy()
        valid = np.arange(nz.shape[0])[:, None] < counts[None, :]
        rows = np.broadcast_to(np.arange(self.count)[None, :], valid.shape)[valid]
        off = coo_matrix((nz['value'][valid], (rows, nz['column_index'][valid])),
                         shape=(self.count, self.count)).tocsc()
        sq = np.zeros((3*self.count, 6))
        sv = np.zeros_like(sq)
        mass_dt = self.mass/dt**2
        # Contact-free A and preconditioner are invariant over all iterations.
        first = solver.linear_records[0][0]
        a = kron(off+diags(as_numpy(first['diag'])), eye(3), format='csc')
        inv_m = as_numpy(first['inv_m']).astype(float)
        for k, records in enumerate(solver.linear_records):
            sq[self.fixed] = ((k+1)*dt*self.boundary)[self.fixed]
            sv[self.fixed] = self.boundary[self.fixed]
            previous = sq.copy()
            inertia = sq+dt*sv
            inertia[self.fixed] = sq[self.fixed]
            for record in records:
                if progress is not None:
                    progress()
                q = as_numpy(record['q']).astype(float)
                force_jac = stretch_force_jacobian(q, self.faces, self.rest,
                                                  self.areas, self.stiffness)+self.bend
                db = mass_dt[:, None]*(inertia-sq)+force_jac@sq
                dx = dt*sv.copy() if record['initial'] else np.zeros_like(sq)
                dx[self.fixed] = 0
                sq += pcg_tangent(a, inv_m, record, db, dx)
            sv = .998*(sq-previous)/dt
            sv[self.fixed] = self.boundary[self.fixed]
            if not np.isfinite(sq).all():
                raise SensitivityUnavailable('nonfinite solver tangent')
            yield sq.reshape(self.count, 3, 6).copy()

def pcg_tangent(a, inv_m, record, db, dx):
    """Six forward-mode directions through upstream's seven PCG operations."""
    dr = db-a@dx
    dp = np.zeros_like(dx)
    old_num, old_dnum, old_p = 0., np.zeros(6), np.zeros(len(dx))
    for i, arrays in enumerate(record['steps']):
        r, z, p, ap = [as_numpy(array).astype(float).ravel() for array in arrays]
        dz = np.einsum('nxy,nyd->nxd', inv_m, dr.reshape(-1, 3, 6)).reshape(-1, 6)
        num = record['rz'][i]
        # Tall, six-column reductions are memory-bound. Avoid threaded BLAS
        # launch/synchronization overhead for these tiny-output operations.
        dnum = np.einsum('i,ij->j', z, dr)+np.einsum('i,ij->j', r, dz)
        beta, dbeta = 0., np.zeros(6)
        if i and abs(old_num) > 1e-30 and np.isfinite([old_num, num]).all():
            beta = num/old_num
            dbeta = (dnum-beta*old_dnum)/old_num
        dp = dz+beta*dp+old_p[:, None]*dbeta
        dap = a@dp
        denom = record['pap'][i]
        ddenom = np.einsum('i,ij->j', ap, dp)+np.einsum('i,ij->j', p, dap)
        alpha, dalpha = 0., np.zeros(6)
        if abs(denom) > 1e-30 and np.isfinite([denom, num]).all():
            alpha = num/denom
            dalpha = (dnum-alpha*ddenom)/denom
        dr -= alpha*dap+ap[:, None]*dalpha
        dx += alpha*dp+p[:, None]*dalpha
        old_num, old_dnum, old_p = num, dnum, p
    return dx


def observation_jacobian(q, sensitivity, runtime, faces=None, weights=None):
    """Capsule distance and optional material-strain chain rules."""
    if faces is not None:
        triangles = runtime.model.tri_indices.numpy()[faces]
        p = np.einsum('ti,tix->tx', weights, q[triangles])
        p += runtime.hit_offsets.numpy()[:len(faces)]
        s = np.einsum('ti,tixd->txd', weights, sensitivity[triangles])
    else:
        section = slice(runtime.particle_start, runtime.particle_start+runtime.particle_count)
        p, s = q[section], sensitivity[section]
    _, normals, _ = runtime.capsule.distances(p, runtime.particle_radius)
    out = np.einsum('tx,txd->td', normals, s)
    if faces is not None or not runtime.settings.strain_enabled:
        return out
    triangles = runtime.model.tri_indices.numpy()
    rest = runtime.model.tri_poses.numpy()
    coeff = np.concatenate((-rest.sum(axis=1)[:, None, :], rest), axis=1)
    f = np.einsum('tix,tia->txa', q[triangles], coeff)
    df = np.einsum('tixd,tia->txad', sensitivity[triangles], coeff)
    lengths = np.linalg.norm(f, axis=1)
    if np.any(lengths <= 1e-6):
        raise SensitivityUnavailable('degenerate strain observation')
    unit = f/lengths[:, None, :]
    dl = np.einsum('txa,txad->tad', unit, df)
    cosine = np.sum(unit[:, :, 0]*unit[:, :, 1], axis=1)
    dc = ((np.einsum('tx,txd->td', unit[:, :, 1], df[:, :, 0])
           -cosine[:, None]*dl[:, 0])/lengths[:, 0, None]
          +(np.einsum('tx,txd->td', unit[:, :, 0], df[:, :, 1])
            -cosine[:, None]*dl[:, 1])/lengths[:, 1, None])
    strain = np.stack((-dl[:, 0], -dl[:, 1], -dc, dc), axis=1)
    return np.concatenate((out, strain.reshape(-1, 6)))

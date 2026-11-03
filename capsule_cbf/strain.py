"""Material-coordinate triangle strain; no Newton dependency."""
import numpy as np


class TriangleStrain:
    def __init__(self, triangles, inverse_rest, weft_limit=.02, warp_limit=.02,
                 shear_limit=.2):
        self.triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
        self.inverse_rest = np.asarray(inverse_rest, dtype=float).reshape(-1, 2, 2)
        self.limits = np.array([weft_limit, warp_limit, shear_limit, shear_limit])
        if len(self.triangles) != len(self.inverse_rest):
            raise ValueError('Triangle/rest matrix count mismatch')
        if (not np.isfinite(self.inverse_rest).all() or
                np.any(abs(np.linalg.det(self.inverse_rest)) < 1e-12) or
                not np.isfinite(self.limits).all() or np.any(self.limits <= 0) or
                shear_limit >= 1):
            raise ValueError('Invalid rest matrices or strain limits')

    def measures(self, points):
        p = np.asarray(points, dtype=float)[self.triangles]
        edges = np.stack((p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=-1)
        f = edges @ self.inverse_rest
        lengths = np.linalg.norm(f, axis=1)
        cosine = np.sum(f[:, :, 0]*f[:, :, 1], axis=1) / np.maximum(
            lengths[:, 0]*lengths[:, 1], 1e-12)
        return np.column_stack((lengths-1., cosine, -cosine))

    def barriers(self, points):
        return self.limits-self.measures(points)

    def aggregate(self, points, subset, kappa=40.):
        """Kreisselmeier-Steinhauser soft maximum of measure/limit and its gradient.

        KS = (1/kappa) log sum exp(kappa g) >= max g, with excess at most
        log(4 T)/kappa; g = (weft, warp, +shear, -shear)/limits. Returns
        (KS, dKS/dpoints with shape (N, 3)). Leading batch axes of `points`
        (e.g. prediction times) are evaluated together.
        """
        p = np.asarray(points, dtype=float)
        batch = p.shape[:-2]
        p = p.reshape(-1, *p.shape[-2:])
        tris, inverse = self.triangles[subset], self.inverse_rest[subset]
        coeff = np.concatenate((-inverse.sum(axis=1)[:, None, :], inverse), axis=1)
        f = np.einsum('btix,tia->btxa', p[:, tris], coeff)
        lengths = np.maximum(np.linalg.norm(f, axis=2), 1e-12)
        unit = f/lengths[:, :, None, :]
        cosine = np.sum(unit[..., 0]*unit[..., 1], axis=2)
        g = np.stack((lengths[..., 0]-1., lengths[..., 1]-1., cosine, -cosine), axis=2)/self.limits
        top = g.max(axis=(1, 2), keepdims=True)
        w = np.exp(kappa*(g-top))
        total = w.sum(axis=(1, 2), keepdims=True)
        value = top[:, 0, 0]+np.log(total[:, 0, 0])/kappa
        w /= total
        du = w[..., 0]/self.limits[0]
        dv = w[..., 1]/self.limits[1]
        ds = (w[..., 2]-w[..., 3])/self.limits[2]
        u, v = unit[..., 0], unit[..., 1]
        grad_fu = du[..., None]*u+ds[..., None]*(v-cosine[..., None]*u)/lengths[..., 0, None]
        grad_fv = dv[..., None]*v+ds[..., None]*(u-cosine[..., None]*v)/lengths[..., 1, None]
        corner = (coeff[None, :, :, 0, None]*grad_fu[:, :, None, :]
                  +coeff[None, :, :, 1, None]*grad_fv[:, :, None, :])
        gradient = np.zeros_like(p)
        for j in range(3):
            np.add.at(gradient, (slice(None), tris[:, j]), corner[:, :, j])
        return value.reshape(batch), gradient.reshape(*batch, *p.shape[-2:])

    def summary(self, points):
        m = self.measures(points)
        return dict(max_weft_strain=float(m[:, 0].max()),
                    max_warp_strain=float(m[:, 1].max()),
                    max_shear_cosine=float(abs(m[:, 2]).max()),
                    min_strain_barrier=float((self.limits-m).min()))

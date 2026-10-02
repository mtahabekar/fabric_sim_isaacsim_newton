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

    def summary(self, points):
        m = self.measures(points)
        return dict(max_weft_strain=float(m[:, 0].max()),
                    max_warp_strain=float(m[:, 1].max()),
                    max_shear_cosine=float(abs(m[:, 2]).max()),
                    min_strain_barrier=float((self.limits-m).min()))

"""Capsule envelope of a box obstacle: signed distance, normals and surface points.

The capsule stands in for the table (or any obstacle it encloses): a vertical
cylinder of radius sqrt(hx^2 + hy^2) around the box, closed below by a
hemisphere and above either by a hemisphere or by a spherical cap through the
four top corners of the box (`centered_sphere`, the default). The cap sphere
lies on the box axis and has radius top_radius_scale * sqrt(hx^2 + hy^2 + hz^2);
its center is lowered so the sphere still passes through the top corners.
Scale 1 is the box-centered sphere (19.0 cm above the top face of the
0.44 x 0.44 x 0.32 m table); the default scale 2 gives a flatter cap, 7.3 cm
above the top face (the hemisphere reaches 31.1 cm). Pure NumPy.
"""
import numpy as np

TOP_CAP_MODES = ('hemisphere', 'centered_sphere')
DEFAULT_TOP_RADIUS_SCALE = 2.0


class Capsule:
    def __init__(self, center, half_extents, top_cap_mode='centered_sphere',
                 top_radius_scale=DEFAULT_TOP_RADIUS_SCALE):
        self.center = np.asarray(center, dtype=float)
        half = np.asarray(half_extents, dtype=float)
        if (self.center.shape != (3,) or half.shape != (3,)
                or not np.isfinite(self.center).all() or not np.isfinite(half).all()
                or np.any(half <= 0)):
            raise ValueError('Expected finite center and positive box half extents')
        if top_cap_mode not in TOP_CAP_MODES:
            raise ValueError(f'top_cap_mode must be one of {TOP_CAP_MODES}')
        self.radius = float(np.linalg.norm(half[:2]))
        self.half_length = float(half[2])
        self.top_cap_mode = top_cap_mode
        self.centered_top = top_cap_mode == 'centered_sphere'
        self.top_radius_scale = float(top_radius_scale)
        if self.centered_top:
            self.top_sphere_radius = self.top_radius_scale*float(np.linalg.norm(half))
            if not np.isfinite(self.top_sphere_radius) or self.top_sphere_radius < self.radius:
                raise ValueError('The top cap sphere must be at least as wide as the box diagonal: '
                                 f'top_radius_scale >= {self.radius/np.linalg.norm(half):.4f}')
            # Height of the sphere center above the capsule center (0 for scale 1).
            rise = np.sqrt(self.top_sphere_radius**2-self.radius**2)
            self.top_center = float(self.half_length-rise)
            self.top_angle = float(np.arctan2(self.radius, rise))
        else:
            self.top_sphere_radius, self.top_center, self.top_angle = self.radius, self.half_length, np.pi/2
        self.meridian_length = (np.pi*self.radius/2+2*self.half_length
                                + self.top_sphere_radius*self.top_angle)

    def surface(self, fractions, angles):
        """Surface points at meridian arc-length fractions (0 bottom, 1 top) and azimuths."""
        s = np.asarray(fractions, dtype=float)*self.meridian_length
        cap = np.pi*self.radius/2
        theta_lo = s/self.radius
        theta_hi = (s-cap-2*self.half_length)/self.radius
        rho = np.where(s < cap, self.radius*np.sin(theta_lo),
                       np.where(s > cap+2*self.half_length, self.radius*np.cos(theta_hi), self.radius))
        z = np.where(s < cap, -self.half_length-self.radius*np.cos(theta_lo),
                     np.where(s > cap+2*self.half_length, self.half_length+self.radius*np.sin(theta_hi),
                              -self.half_length+s-cap))
        if self.centered_top:
            top = s > cap+2*self.half_length
            angle = self.top_angle-(s-cap-2*self.half_length)/self.top_sphere_radius
            rho = np.where(top, self.top_sphere_radius*np.sin(angle), rho)
            z = np.where(top, self.top_center+self.top_sphere_radius*np.cos(angle), z)
        a = np.asarray(angles, dtype=float)
        return (np.stack(np.broadcast_arrays(rho[:, None]*np.cos(a), rho[:, None]*np.sin(a), z[:, None]),
                         axis=-1).reshape(-1, 3)+self.center)

    def distances(self, points, particle_radius=0.):
        """Signed clearance (minus particle_radius), outward unit normals and nearest surface points."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.centered_top:
            return self._centered_top_distances(p, particle_radius)
        axis = np.broadcast_to(self.center, p.shape).copy()
        axis[:, 2] = np.clip(p[:, 2], self.center[2]-self.half_length, self.center[2]+self.half_length)
        delta = p-axis
        lengths = np.linalg.norm(delta, axis=1)
        normals = delta/np.maximum(lengths[:, None], 1e-12)
        normals[lengths < 1e-12] = [1., 0., 0.]  # deterministic subgradient on the axis
        surface = axis+self.radius*normals
        return lengths-self.radius-particle_radius, normals, surface

    def _centered_top_distances(self, points, particle_radius):
        """Exact signed distance to the revolved, piecewise circular profile.

        Top: on-axis sphere through the top box corners (center at
        top_center), trimmed at the top-face plane. Sides and bottom: cylinder
        and lower hemisphere. The rim is a crease.
        """
        p = points-self.center
        rho = np.linalg.norm(p[:, :2], axis=1)
        z, r, h, radius, zc = p[:, 2], self.radius, self.half_length, self.top_sphere_radius, self.top_center
        radial = p[:, :2]/np.maximum(rho[:, None], 1e-12)
        radial[rho < 1e-12] = [1., 0.]
        length_bottom = np.hypot(rho, z+h)
        bottom_r = np.where(z < -h, r*rho/np.maximum(length_bottom, 1e-12), r)
        bottom_z = np.where(z < -h, -h+r*(z+h)/np.maximum(length_bottom, 1e-12), -h)
        length_top = np.hypot(rho, z-zc)
        on_top_arc = (z-zc)/np.maximum(length_top, 1e-12) >= (h-zc)/radius
        top_r = np.where(on_top_arc, radius*rho/np.maximum(length_top, 1e-12), r)
        top_z = np.where(on_top_arc, zc+radius*(z-zc)/np.maximum(length_top, 1e-12), h)
        candidates = np.stack((np.column_stack((bottom_r, bottom_z)),
                               np.column_stack((np.full_like(z, r), np.clip(z, -h, h))),
                               np.column_stack((top_r, top_z))), axis=1)
        profile = np.column_stack((rho, z))
        squared = np.sum((profile[:, None]-candidates)**2, axis=2)
        chosen = np.argmin(squared, axis=1)
        nearest = candidates[np.arange(len(p)), chosen]
        distance = np.sqrt(np.min(squared, axis=1))
        inside = ((rho <= r) & (z >= -h-np.sqrt(np.maximum(r*r-rho*rho, 0)))
                  & (z <= zc+np.sqrt(np.maximum(radius*radius-rho*rho, 0))))
        sign = np.where(inside, -1., 1.)
        normal2 = sign[:, None]*(profile-nearest)/np.maximum(distance[:, None], 1e-12)
        # Surface points need the corresponding outward normal, not 0/0.
        exact = distance < 1e-12
        analytic = np.column_stack((np.ones_like(z), np.zeros_like(z)))
        analytic[chosen == 0] = np.column_stack((nearest[:, 0]/r, (nearest[:, 1]+h)/r))[chosen == 0]
        analytic[chosen == 2] = (nearest[chosen == 2]-[0., zc])/radius
        normal2[exact] = analytic[exact]
        normals = np.column_stack((normal2[:, :1]*radial, normal2[:, 1]))
        surface = self.center+np.column_stack((nearest[:, :1]*radial, nearest[:, 1]))
        return sign*distance-particle_radius, normals, surface

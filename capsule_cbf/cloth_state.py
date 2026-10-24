"""Full-particle kinematic state, independent of Isaac, Warp and ROS.

Dynamics is the coupled evolution law; positions and velocities are its state.
Do not interpret Newton's particle_f external-force buffer as total net force.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ClothState:
    positions: np.ndarray  # (N, 3), world metres
    velocities: np.ndarray  # (N, 3), world metres/second
    particle_start: int = 0  # local row i corresponds to Newton particle start+i

    def __post_init__(self):
        q, v = (np.array(a, dtype=np.float64, copy=True)
                for a in (self.positions, self.velocities))
        if (q.ndim != 2 or q.shape[1] != 3 or v.shape != q.shape
                or not np.isfinite(q).all() or not np.isfinite(v).all()):
            raise ValueError('ClothState requires finite matching (N, 3) positions and velocities')
        if not isinstance(self.particle_start, (int, np.integer)) or self.particle_start < 0:
            raise ValueError('particle_start must be a nonnegative integer')
        q.flags.writeable = v.flags.writeable = False
        object.__setattr__(self, 'positions', q)
        object.__setattr__(self, 'velocities', v)

    @classmethod
    def from_newton(cls, state, start=0, count=None):
        """Read both arrays at the same simulation boundary; no physics advances."""
        q, v = state.particle_q.numpy(), state.particle_qd.numpy()
        if count is None:
            count = len(q)-start
        if (not isinstance(start, (int, np.integer)) or start < 0
                or not isinstance(count, (int, np.integer)) or count < 0
                or start+count > len(q) or len(v) != len(q)):
            raise ValueError('Invalid cloth particle slice')
        return cls(q[start:start+count], v[start:start+count], start)

    def clearance_rates(self, capsule):
        """Instantaneous dh/dt = normal dot velocity for a static capsule.

        This is measured drift, not a derivative with respect to new commands.
        """
        _, normals, _ = capsule.distances(self.positions)
        return np.einsum('ij,ij->i', normals, self.velocities)

    def summary(self, capsule):
        rates = self.clearance_rates(capsule)
        return dict(state_velocity_available=True, state_particle_count=len(rates),
                    min_clearance_rate=float(rates.min()) if len(rates) else None,
                    approaching_particle_count=int(np.count_nonzero(rates < 0)),
                    particle_max_speed=float(np.linalg.norm(self.velocities, axis=1).max(initial=0)))


def unpack_cloth_state(value, capsule):
    """Keep position-only test/custom predictors compatible without inventing v."""
    if isinstance(value, ClothState):
        return value.positions, value.summary(capsule)
    return value, dict(state_velocity_available=False)


def particle_state_payload(positions, velocities=None):
    """Pack little-endian float32 PointCloud2 rows, without importing ROS."""
    q = np.asarray(positions)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError('Particle positions must have shape (N, 3)')
    names = ('x', 'y', 'z')
    if velocities is not None:
        v = np.asarray(velocities)
        if v.shape != q.shape:
            raise ValueError('Particle velocities must match positions')
        q = np.concatenate((q, v), axis=1)
        names += ('vx', 'vy', 'vz')
    return names, np.ascontiguousarray(q, dtype='<f4')

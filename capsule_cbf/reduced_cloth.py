"""Fine-mesh sensing with coarse-mesh linearized dynamics.

Built once from the live Newton model; afterwards pure NumPy/SciPy. Coarse
nodes are a material-coordinate subset of the fine grid (MaterialGrid), so
restriction is an index gather and lifting is barycentric interpolation:
    p_fine(t) = p_fine(0) + W [q_coarse(t) - q_coarse(0)],  dp/du = W dq/du.
"""
import numpy as np
from scipy.sparse import csr_matrix, identity
from .cloth_model import ClothMechanics, HorizonModel


class ReducedCloth:
    def __init__(self, model, patch, corner_ids, offsets, frame_dt, *, particle_start=0,
                 particle_count=None, particle_radius=.008, cells=14, substeps=1, stride=1,
                 plant_substeps=8, jacobian='affine'):
        """stride: plant frames per prediction output; substeps: model steps per output."""
        n = particle_count or model.particle_count-particle_start
        patch = np.asarray(patch, dtype=int)
        corner_ids = np.asarray(corner_ids, dtype=int)
        offsets = np.asarray(offsets, dtype=float).reshape(-1, 3)
        self.fine = ClothMechanics.from_newton(model, particle_start, n)
        self.fine_patch = patch-particle_start
        self.fine_corner_ids = corner_ids
        rest = model.particle_q.numpy()[particle_start:particle_start+n].astype(float)
        self.rest = rest
        if cells:
            from .coarse_prediction import MaterialGrid, build_coarse_model
            grid = MaterialGrid(rest, self.fine_patch, corner_ids, cells)
            coarse, nodes, pin_corners, pin_offsets = build_coarse_model(
                model, grid, patch, corner_ids, offsets, particle_start, particle_radius)
            self.coarse = ClothMechanics.from_newton(coarse)
            self.source = grid.source.astype(int)
            faces, weights = grid.locate(rest[:, :2])
            cols = grid.triangles[faces]
            self.lift = csr_matrix((weights.astype(float).ravel(),
                                    (np.repeat(np.arange(n), 3), cols.ravel())),
                                   shape=(n, len(self.source)))
            self.grid_shape = [len(grid.x)-1, len(grid.y)-1]
        else:
            self.coarse = self.fine
            self.source = np.arange(n)
            nodes, pin_corners, pin_offsets = self.fine_patch, corner_ids, offsets
            self.lift = identity(n, format='csr')
            self.grid_shape = None
        self.pin_nodes = np.asarray(nodes, dtype=int)
        self.pin_corners = np.asarray(pin_corners, dtype=int)
        self.stride = stride
        self.horizon = HorizonModel(self.coarse, self.pin_nodes, self.pin_corners, pin_offsets,
                                    stride*frame_dt, substeps, plant_dt=frame_dt/plant_substeps,
                                    jacobian=jacobian)
        self.info = dict(prediction_model='coarse' if cells else 'full', live_particles=n,
                         prediction_particles=self.coarse.count,
                         prediction_triangles=len(self.coarse.triangles),
                         prediction_grid_shape=self.grid_shape, model_substeps=substeps,
                         stride=stride, jacobian=jacobian)

    def restrict(self, fine_values):
        return np.asarray(fine_values, dtype=float)[self.source]

    def lift_points(self, indices, fine_q0, coarse_q0, coarse_q):
        """Anchored fine positions p_i(0) + W_i (q_c - q_c0) for selected points."""
        w = self.lift[indices]
        return fine_q0[indices]+(w@(coarse_q-coarse_q0).reshape(len(coarse_q0), -1)).reshape(-1, 3)

    def lift_sensitivity(self, indices, sensitivity):
        """W_i S for selected fine points, sensitivity shape (coarse, 3, C)."""
        s = np.asarray(sensitivity)
        return (self.lift[indices]@s.reshape(len(s), -1)).reshape(len(indices), *s.shape[1:])

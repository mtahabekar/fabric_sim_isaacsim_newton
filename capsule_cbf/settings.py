"""Settings of the sensitivity CBF; defaults are the validated configuration.

capsule_cbf/config.json lists the same values. Sections and numbers refer to
SENSITIVITY_CBF_REPORT.md.
"""
from dataclasses import dataclass, fields
import json
from pathlib import Path
import numpy as np
from .capsule import DEFAULT_TOP_RADIUS_SCALE, TOP_CAP_MODES

CONTROLLERS = ('sensitivity', 'none')
MODEL_JACOBIANS = ('affine', 'trajectory', 'frozen')


@dataclass
class Settings:
    # 'sensitivity' filters the followers; 'none' applies the nominal command
    # unfiltered but still measures clearance, tension and strain (baseline runs).
    controller: str = 'sensitivity'
    # Obstacle envelope and its USD wireframe (sectors x bands lines).
    top_cap_mode: str = 'centered_sphere'
    # centered_sphere cap: sphere radius in units of the box half diagonal; the
    # sphere always passes through the four top corners. 1 = box-centered sphere
    # (19 cm above the table top), 2 = flatter cap (7.3 cm).
    top_cap_radius_scale: float = DEFAULT_TOP_RADIUS_SCALE
    sectors: int = 24
    bands: int = 18
    margin: float = 0.05  # m of clearance kept from the capsule (h = d - margin)
    # Nominal command: leader velocity plus formation recovery toward the
    # initial follower-leader offsets (horizontal only unless enabled).
    formation_gain: float = 0.5
    vertical_formation_recovery: bool = False
    max_speed: float = 0.5  # m/s per follower; zero disables the bound
    follower_max_accel: float = 3.0  # m/s^2 per axis; zero disables the rate box
    command_smoothing: float = 0.25  # weight of the previous command in the QP objective
    # Extra relaxation (row units: 1 N tension, 5 % of a strain limit) of an
    # infeasible stretch group, so the command stays near nominal instead of
    # being pinned to the rate-box vertex of the least-violation LP (qp.py).
    hard_relaxation_tolerance: float = 1.0
    # Prediction model (section 5): coarse grid, horizon, linearization.
    prediction_grid_cells: int = 14  # base cells per material axis; 0 predicts on the full mesh
    horizon_frames: int = 20  # plant frames covered by the prediction (0.33 s at 60 Hz)
    horizon_stride: int = 2  # plant frames per prediction output
    model_substeps: int = 1
    model_jacobian: str = 'affine'
    warmup_frames: int = 90  # the spawned flat sheet is slack; filter starts once it hangs
    # Clearance rows (section 6).
    alpha: float = 2.0  # 1/s decay rate of the discrete-time barrier condition
    prediction_activation: float = 0.0  # rows for points whose predicted h reaches this
    candidate_rings: int = 3  # rest-mesh rings defining a local clearance minimum
    max_candidates: int = 12
    terminal_lookahead: float = 0.15  # s; terminal row h + tau dh/dt >= 0
    # Stretch rows: hard group, priority over clearance.
    tension_limit: float = 20.0  # N per gripper; zero disables
    strain_enabled: bool = True
    weft_strain_limit: float = 0.02  # policy limits, not measured failure strains
    warp_strain_limit: float = 0.02
    shear_limit: float = 0.20  # absolute cosine of the material-axis angle
    strain_band_radius: float = 0.06  # m around each gripper patch (rest frame)
    ks_kappa: float = 40.0
    stretch_alpha: float = 4.0
    # Robustness (section 7).
    leader_max_accel: float = 0.3  # m/s^2 per axis, worst-case leader deviation
    conformal_quantile: float = 0.95
    conformal_window: int = 600
    initial_clearance_margin: float = 0.002  # m, before residual data exist
    initial_tension_margin: float = 0.5  # N
    initial_strain_margin: float = 0.05  # fraction of the strain limit
    observer_gain: float = 0.3

    def __post_init__(self):
        if self.controller not in CONTROLLERS:
            raise ValueError(f'controller must be one of {CONTROLLERS}')
        if self.top_cap_mode not in TOP_CAP_MODES:
            raise ValueError(f'top_cap_mode must be one of {TOP_CAP_MODES}')
        if self.model_jacobian not in MODEL_JACOBIANS:
            raise ValueError(f'model_jacobian must be one of {MODEL_JACOBIANS}')
        integers = ('sectors', 'bands', 'prediction_grid_cells', 'horizon_frames', 'horizon_stride',
                    'model_substeps', 'warmup_frames', 'candidate_rings', 'max_candidates', 'conformal_window')
        for name in integers:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('horizon_frames', 'horizon_stride', 'model_substeps', 'candidate_rings',
                     'max_candidates', 'conformal_window'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.prediction_grid_cells == 1:
            raise ValueError('prediction_grid_cells must be 0 or at least 2')
        if self.sectors < 3 or self.bands < 2:
            raise ValueError('The capsule wireframe needs at least 3 sectors and 2 bands')
        if self.horizon_frames % self.horizon_stride:
            raise ValueError('horizon_frames must be a multiple of horizon_stride')
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is float and (not np.isfinite(value) or value < 0):
                raise ValueError(f'{f.name} must be finite and nonnegative')
        if min(self.weft_strain_limit, self.warp_strain_limit, self.shear_limit) <= 0 or self.shear_limit >= 1:
            raise ValueError('Strain limits must be positive and shear_limit below 1')
        if not 0 < self.conformal_quantile < 1 or self.ks_kappa == 0:
            raise ValueError('conformal_quantile must lie in (0, 1) and ks_kappa be positive')


def load_settings(path):
    """Settings from a JSON file; unknown keys are rejected."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(data)-known)
    if unknown:
        raise ValueError(f'Unknown setting(s) in {path}: {unknown}')
    return Settings(**data)

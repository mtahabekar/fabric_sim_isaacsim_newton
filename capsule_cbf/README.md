# capsule_cbf — the sensitivity CBF

Safety filter for the two follower grippers of a cloth held at four corners.
Every frame it measures the full cloth, linearizes a coarse cloth model once,
predicts 0.33 s ahead together with the exact sensitivity of every predicted
point to the six follower velocity components, and solves one small QP that
enforces, at every predicted time:

- clearance from a capsule enclosing the obstacle (rest-mesh local minima whose
  predicted barrier reaches zero within the horizon, plus a terminal look-ahead row);
- gripper tension and KS-aggregated weft/warp/shear strain near each gripper
  (hard group: priority over clearance);
- follower speed and acceleration limits.

Margins are tightened online by conformal quantiles of measured prediction
residuals, a disturbance observer corrects the model, and infeasibility is
resolved lexicographically (stretch first). The library is pure NumPy/SciPy
at run time; Newton is used only to build the model. The formulation,
alternatives, validation and results are in
[SENSITIVITY_CBF_REPORT.md](SENSITIVITY_CBF_REPORT.md).

## Modules

| File | Contents |
| --- | --- |
| `sensitivity_cbf.py` | `SensitivityCbfController.update(cloth_state, corners, leaders, dt, nominal=None) -> (command (2, 3), diagnostics)` |
| `settings.py`, `config.json` | `Settings` (every parameter documented) and `load_settings`; `config.json` holds the validated values |
| `reduced_cloth.py` | Fine-mesh sensing with a coarse prediction mesh (restriction and barycentric lifting) |
| `cloth_model.py` | Style3D-consistent forces, exact and PSD-projected stiffness, gripper resultants, linearized horizon model with sensitivities |
| `material_grid.py` | Coarse material grid and its Newton model (same density, material and obstacles) |
| `qp.py` | Exact least-distance QP (NNLS dual) with lexicographic hard/soft relaxation |
| `strain.py` | Triangle strain measures and their KS aggregate with gradient |
| `capsule.py` | Capsule envelope: signed distance, normals, surface points |
| `cloth_state.py` | `ClothState` (positions and velocities of every particle) and the PointCloud2 payload |
| `newton_runtime.py` | `ClothRuntime`: Newton Style3D stepping with prescribed gripper patches (live cloth and headless plant) |
| `scenario.py` | Headless Newton plant with the Isaac configuration |
| `benchmark.py` | Closed-loop scenarios (descend, pull, descend_pull) with and without the filter |
| `validate_sensitivity.py` | Model and sensitivity accuracy against central differences of the nonlinear plant |
| `plot_report.py` | Report figures from `results/` |
| `results/`, `figures/` | Data and figures cited by the report |

## Use

```python
from capsule_cbf.settings import load_settings
from capsule_cbf.reduced_cloth import ReducedCloth
from capsule_cbf.sensitivity_cbf import SensitivityCbfController

cfg = load_settings('capsule_cbf/config.json')
reduced = ReducedCloth(model, patch, patch_corners, patch_offsets, frame_dt, particle_start=start,
                       particle_count=count, cells=cfg.prediction_grid_cells, stride=cfg.horizon_stride)
ctrl = SensitivityCbfController(capsule, reduced, rest_corners, particle_radius, cfg, frame_dt)
command, debug = ctrl.update(ClothState(q, v), corners, leaders, frame_dt)
```

`fabric_newton_isaacsim.py` (repository root) does exactly this inside Isaac
Sim. The followers start filtering after `warmup_frames`, once the spawned
flat sheet hangs: the affine model is linearized about tensioned cloth.
`controller: "none"` returns the nominal command unfiltered while still
measuring clearance, tension and strain, for baseline runs.

The diagnostics `status` is `feasible`, `relaxed_soft` (clearance rows relaxed),
`relaxed_hard` (stretch rows relaxed), `solver_failure` or `unfiltered`.
A relaxed status means the requested limits could not all be met; the filter
is a linearized predictive filter with empirical margins, not a certificate.

## Validated operating range

Validated on the 56 x 56 cloth (0.2 kg/m^2) with leaders moving at 0.1 m/s
(report, section 10). Outside it the stretch group can become infeasible and,
because stretch has priority, clearance yields:

- leaders descending below the capsule top (e.g. 0.40 m from z = 0.85 m) make
  clearance and the stretch limits physically incompatible;
- leader speeds of 0.2-0.4 m/s: stretch rows infeasible in most frames, margin
  entered by about 3-4 cm (headless plant) up to contact (Isaac, 0.4 m/s);
- coarse cloths (20 x 20): predicted stretch rows grossly infeasible, contact.

## Tests and tools

Run from the repository root in an environment with `newton==1.6.1`,
`warp-lang==1.17.0`, `numpy`, `scipy` (Newton-dependent tests skip without it).
Isaac Sim 6.2's own interpreter and libraries work without starting Kit:

```bash
ISAAC=/media/taha/ExtermeLinux/isaacsim6.2
export PYTHONPATH=$PWD:$ISAAC/extscache/omni.warp.core-1.17.0+lx64:$ISAAC/exts/isaacsim.pip.newton/pip_prebundle:$ISAAC/exts/omni.pip.compute/pip_prebundle:$(ls -d $ISAAC/extscache/omni.kit.pip_archive-*/pip_prebundle)
alias python=$ISAAC/kit/python/bin/python3
```

```bash
export PYTHONPATH=$PWD:$PYTHONPATH OPENBLAS_NUM_THREADS=1
python -m unittest discover -s capsule_cbf/tests -t .        # 51 tests
python -m capsule_cbf.benchmark --output /tmp/bench           # nominal vs sensitivity, 3 scenarios
python -m capsule_cbf.validate_sensitivity --output /tmp/sens.json [--all-variants]
python -m capsule_cbf.plot_report --results capsule_cbf/results --output /tmp/figures   # needs matplotlib
```

Use one BLAS thread (`OPENBLAS_NUM_THREADS=1`): the filter's many small
LAPACK calls are several times slower with a thread pool.

# Coarse cloth dynamics for the CBF

The capsule launch now predicts with a **289-particle cloth**, while live physics
and rendering still use **3,249 particles**. Both dynamics and all six command
derivatives run on the coarse mesh on CUDA. Nonlinear corrected-command checks
also use this coarse model; ordinary control does not run a hidden fine predictor.

`prediction_grid_cells: 14` requests 14 base cells per material axis. Additional
coordinate lines preserve the physical boundaries of the existing gripper
patches. For the default live mesh this yields 16-by-16 cells, 17-by-17 particles,
and 512 triangles. The live mesh remains 56-by-56 cells and 6,272 triangles.
Actual counts are printed at startup and included in CBF diagnostics.

## How it works

1. **Map in material coordinates.** Coarse nodes are a subset of the initial
   planar cloth grid. Each material point maps to a coarse triangle and
   barycentric weights. The association uses rest XY coordinates, so folds
   cannot make a point jump to an unrelated nearby part of the fabric.

2. **Preserve the physical cloth parameters.** The coarse model retains rest
   dimensions, gravity, particle radius, anisotropic material parameters, static
   obstacles, and contact settings. Grid lines at gripper patch boundaries
   preserve their physical size. Density is `sum(fine_particle_mass)/rest_area`;
   the mesh builder distributes it by triangle area. Reusing the fine mass per
   vertex would incorrectly make the coarse cloth lighter. Coarse rest matrices
   and bending data are rebuilt from its topology.

3. **Refresh the measured state each frame.** A Warp gather copies positions
   **and velocities** at retained fine nodes into a private coarse snapshot.
   The predictor never evolves independently across frames. Public `ClothState`
   still contains every fine particle's position and velocity for diagnostics
   and other uses; only the prediction dynamics are reduced.

4. **Detect proximity on the actual fine mesh.** The existing ray/BVH detector
   and basin selection still use the fine live cloth. This is much cheaper than
   repeatedly solving its dynamics and preserves detection of current wrinkles.
   Selected fine-triangle hits are mapped into coarse material triangles.

5. **Anchor each predicted hit to its measured position.** If `W` interpolates
   a selected point from coarse vertices, define

   ```text
   residual = p_fine(0) - W q_coarse(0)
   p_predicted(t) = W q_coarse(t) + residual
                  = p_fine(0) + W [q_coarse(t) - q_coarse(0)]
   ```

   Starting distances therefore match the actual fine surface, including
   wrinkles omitted by the coarse geometry. A GPU kernel computes the residual.
   It remains fixed over this one-frame horizon and is refreshed from the next
   live snapshot. Fine motion between retained nodes is approximated by coarse
   displacement; sampled velocities drive the coarse dynamics.

6. **Differentiate the coarse solver.** The existing inline Warp backend
   propagates six follower-velocity derivatives through every coarse particle.
   The snapshot and residual are independent of the new command, so

   ```text
   dp_predicted/du = W dq_coarse/du
   dh/du = normal(p_predicted) dot dp_predicted/du
   ```

   Nonlinear distances and derivatives use the same offset-corrected point.
   Using the normal at the uncorrected coarse point would be inconsistent.

7. **Optimize and check commands on the coarse model.** All existing substep
   constraints and nonlinear command checks remain. Contact/invalid derivatives
   trigger finite differences on the coarse predictor. The resulting command
   is applied to ordinary fine live physics. Diagnostics identify this as
   `prediction_model: coarse` and explicitly describe its approximate scope.

`final_points()` lifts coarse displacement onto the fine snapshot for diagnostics;
it does not execute fine physics. Selected-hit observations use direct
interpolation at their coarse material coordinates.

## GPU measurements

On the host RTX 5070 Laptop GPU, with 8 substeps, 10 nonlinear iterations, 16 PCG
iterations, and five active-CBF recovery frames:

| Predictor | Prediction particles | Median control + live physics |
| --- | ---: | ---: |
| Previous full-resolution inline GPU | 3,249 | 279.9 ms |
| Coarse inline GPU | 289 | **178.2 ms** |

This saves about **36%** of total frame time (about **1.57 times faster**). The
coarse p95 was 203.8 ms, corresponding to roughly 5.6 updates/second at the median.
These are headless timings, excluding rendering, ROS, startup compilation, and
the independent benchmark-only audit rollout.

Removing 91% of prediction particles does not remove 91% of frame time: substep,
nonlinear iteration, PCG iteration, reduction and kernel-launch counts remain
unchanged. Three prediction trajectories were still needed in each active frame,
and live physics remains full resolution. No timestep, iteration count, margin,
or constraint check was reduced.

For this five-frame run:

- Maximum one-frame position error anywhere on the lifted cloth: **6.94 mm**.
- Position RMS error per frame: approximately **0.41–0.51 mm**.
- Maximum final-substep barrier error at selected fine material hits: **0.071 mm**.
- Maximum coarse Jacobian discrepancy versus central differences: `1.55e-5 s`.
- Held-out coarse barrier linearization error: `2.04e-7 m`.
- Prediction left the live state untouched.

The cloth started at z=0.55 m, already inside the clearance margin; all five
recovery frames remained `prediction_not_safe`. This measures computation and
approximation, not successful obstacle avoidance. Selected-hit error was measured
at the final substep, not at every intermediate substep. Raw reports are
`/tmp/capsule-coarse-gpu-56.json` and `/tmp/capsule-inline-gpu-56.json`.

Small wrinkles, high-frequency velocities and fine contacts can evolve differently
between meshes. The sample errors and existing 0.1 mm clearance tightening are
not a proven model-error bound. Coarse feasibility does not certify fine-mesh
safety. Use full-resolution prediction when this approximation is unacceptable.

## Configuration and reproduction

The launch defaults in `capsule_cbf/config.json` are:

```json
"prediction_grid_cells": 14,
"sensitivity_method": "solver_tangent",
"sensitivity_backend": "warp"
```

Set resolution to `0` to restore the previous full predictor. Integers `>= 2`
request base cells per material axis, capped at the live grid size; gripper
boundary lines are added automatically.

The coarse path currently supports one initially flat XY rectangular cloth with
uniform density/material, four corner gripper patches, and static plane/box
obstacles. It requires `ray_basins_enabled: true` and `strain_enabled: false`.
For full strain control set resolution to `0`. Unsupported configurations fail
explicitly rather than silently changing material or obstacle properties.

Launch normally:

```bash
python3 capsule_cbf/launch_capsule_sim.py
```

With the Isaac Python environment from [GPU_SENSITIVITY.md](GPU_SENSITIVITY.md),
reproduce the benchmark:

```bash
/media/taha/ExtermeLinux/isaacsim/python.sh -m capsule_cbf.validate_newton \
  --device cuda:0 --grid 56 --prediction-grid-cells 14 \
  --substeps 8 --frames 5 --settle-frames 0 --cloth-z .55 \
  --output /tmp/capsule-coarse-gpu-56.json
```

Repeat with `--prediction-grid-cells 0` and another output filename for the full
predictor. The validator advances the fine live solver to compare the actual
result with the returned coarse command prediction.

## Changed files and verification

All **57 tests passed** in the Isaac Python runtime with host CUDA access,
including the coarse and full-resolution CUDA graph tests. `git diff --check`
also passed.

- `coarse_prediction.py`: material maps, model construction, GPU state transfer,
  selected-hit anchoring and diagnostic displacement lifting.
- `newton_runtime.py`, `warp_sensitivity.py`, `dynamics_sensitivity.py`: consistent
  offsets in nonlinear observations and both derivative backends.
- `fabric_newton_isaacsim_capsule.py`: predictor selection and resolution logging.
- `core.py`, `config.json`: validated resolution option and launch default.
- `ray_basins.py`: explicit coarse approximation scope in diagnostics.
- `validate_newton.py`: separate prediction/live resolutions and error metrics.
- `tests/test_coarse_prediction.py`: mapping, mass/obstacles, gripper boundaries,
  q/v transfer, hit anchoring, frame resets, isolation, derivatives, CPU reference,
  identity-resolution equivalence, and actual CUDA graph execution.

The installed Style3D solver is unchanged. Interactive Isaac/ROS rendering was
not rerun for this change; host CUDA tests and headless dynamics benchmarks ran.

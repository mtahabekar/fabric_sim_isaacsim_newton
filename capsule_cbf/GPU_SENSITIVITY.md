# Inline GPU cloth derivatives

The default predictor now propagates all six follower-command derivatives in
Warp kernels alongside the ordinary Style3D solver. It does not record or replay
the solver trajectory. This applies to **every particle in the prediction mesh**,
including the interior. The launch now uses a coarse mesh; see
[COARSE_PREDICTION.md](COARSE_PREDICTION.md) for mapping and newer benchmarks.
The full-resolution measurements below describe `prediction_grid_cells: 0`.
The implementation is in `warp_sensitivity.py`; the installed Newton sources are
not edited. `TangentStyle3D` and `TangentPcg` subclass the installed solver and
retain its original primal operations.

## Why the window stalled

The previous analytic derivative path simulated on CUDA, but downloaded solver
checkpoints and replayed derivatives using NumPy/SciPy on the main thread. The
original recorder also allocated and read back individual iteration buffers.
At 8 substeps, 10 nonlinear iterations, and 16 PCG iterations there are **1,280
PCG iterations per prediction**. Four recorded vector arrays alone occupy
approximately 190 MiB for 3,249 particles. Repeated synchronization, transfers,
and CPU work blocked the same thread that runs Isaac's UI.

The retained CPU reference now preallocates its checkpoints and batches transfers,
but its derivative replay is still expensive. The new default eliminates that
replay and the checkpoint tape entirely. CUDA graph capture remains enabled.
The UI is also given a chance to update between prediction trajectories.

## What each step does

1. **Start from the full measured state.** `NewtonPredictor.begin()` snapshots
   positions and velocities. Prediction cannot advance the live simulation.
   The new command has not affected that initial state, so its derivatives start
   at zero. Device arrays `sq` and `sv` have shape `(6, N)` with `vec3` entries,
   equivalent to `(N, 3, 6)` mathematically.

2. **Seed the prescribed gripper motion.** The six columns are the XYZ velocity
   components of follower 0 and follower 1. At time `t`, each particle in a
   follower patch has `dq/du = t I` in its three command columns and `dv/du = I`.
   Leader derivatives are zero because their commands are held fixed. Kernels
   restore these exact boundary values at each substep.

3. **Propagate inertia.** For free particles,
   `Dq_inertia = Dq_previous + dt Dv_previous`. Current measured velocities also
   influence the nominal trajectory, even though their initial command
   derivatives are zero. Measured velocity and command sensitivity answer
   different questions; one does not replace the other.

4. **Differentiate material forces on device.** Triangle kernels evaluate the
   directional derivatives of the installed stretch and normalized shear force
   formulas. Edge kernels differentiate bending. They accumulate six
   Jacobian-vector products directly, without building a dense cloth Jacobian.
   The derivative right-hand side is
   `Db = M/dt² (Dq_inertia - Dq) + Df_stretch + Df_bend`.

5. **Differentiate each actual PCG iteration.** The contact-free projective
   dynamics matrix and preconditioner are constant with respect to this command.
   Consequently `D(Ap) = A Dp` and `Dz = inverse_preconditioner Dr`. However,
   PCG's scalar coefficients depend on the right-hand side and must also be
   differentiated. For example:

   ```text
   n = rᵀz              Dn = Drᵀz + rᵀDz
   d = pᵀAp             Dd = DpᵀAp + pᵀD(Ap)
   alpha = n/d          Dalpha = (Dn - alpha Dd)/d
   Dx += alpha Dp + Dalpha p
   Dr -= alpha D(Ap) + Dalpha Ap
   ```

   The corresponding beta derivative updates the search direction. Six native
   device reductions produce each set of scalar derivatives. Only one previous
   primal search direction is retained in a reusable buffer; no history is
   recorded. The same denominator guards as upstream PCG are used.

6. **Update particle motion.** Each nonlinear increment updates `Dq`. At the
   end of a substep, `Dv = 0.998 (Dq_new - Dq_previous)/dt`, matching installed
   Style3D damping. The next substep continues with these derivatives. This
   differentiates the finite sequence the solver actually executes, rather than
   assuming it reached a converged implicit solution.

7. **Compute barrier derivatives on device.** For a static obstacle,
   `Dh = normalᵀ Dq`. Selected ray hits use barycentric interpolation of the
   triangle vertices and their derivatives. Optional strain observations also
   have Warp derivative kernels. Values are stored for every substep.

8. **Return the compact result to the controller.** `sensitivity()` checks the
   contact/invalid flags, then copies observation Jacobians. It performs no
   derivative replay and launches no additional physics trajectory. Basin
   selection and the small command optimization remain on CPU. Nonlinear
   prediction still checks corrected commands before returning them.

The derivative workspace scales as `O(6N)`, plus observation outputs, rather than
with the number of solver iterations. At the default resolution with strain
disabled it occupies approximately 3.2 MiB including observations. There are no
per-iteration host readbacks, checkpoint clones, or derivative allocations.

## Measured on the real GPU

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GiB, driver 580.178.04; Newton
1.2.1 and Warp 1.13.0. Sequential headless runs used 56-by-56 cloth cells
(3,249 particles, 6,272 triangles), 8 substeps, 10 nonlinear iterations and 16
PCG iterations. The cloth started at z=0.55 m, with the centered spherical cap.
Both methods used the same physical and controller settings and CUDA graphs.

The table uses the median of the first three active-CBF frames in each run:

| Work per control frame | CPU reference derivatives | Inline Warp derivatives |
| --- | ---: | ---: |
| Three prediction trajectories | 105.5 ms | 250.5 ms, including derivatives |
| Two derivative requests after prediction | 4,901.5 ms | 0.24 ms, extraction only |
| Command optimization | 3.88 ms | 1.72 ms |
| Live physics | 25.50 ms | 24.90 ms |
| Total control + live physics | **5,039.4 ms** | **279.7 ms** |

This is about **18 times faster** than the CPU reference. All five inline frames
gave a median of 279.9 ms and p95 of 309.5 ms. It is still only about 3.6 control
updates per wall-clock second in this demanding active-CBF case. Timings exclude
rendering, ROS, startup compilation, and an additional independent validation
rollout used only by the benchmark. Do not interpret extraction time as the
total GPU derivative cost: that computation is included in prediction time.

Three predictions still occur when both correction iterations need nonlinear
validation. The improvement removes six perturbation rollouts **per Jacobian**;
it does not remove these command-validation predictions. Remaining performance
work should target GPU prediction cost and avoid computing derivatives for a
final validation that cannot be relinearized. Reducing mesh or solver resolution
would change the simulation and was not done here.

Raw reports are `/tmp/capsule-inline-gpu-56.json` and
`/tmp/capsule-reference-gpu-56.json` (temporary local files). The test starts
inside the requested clearance margin: all reported recovery frames remain
`prediction_not_safe`. These runs measure computation and derivative agreement;
they are not evidence of successful safety recovery.

## Verification and limits

- All **52 tests passed** using the Isaac Python runtime with host CUDA access;
  the CUDA tests ran rather than being skipped. `git diff --check` also passed.
- Full-resolution inline prediction left the live state untouched. Maximum
  live/predicted position difference was approximately `1.2e-6 m`.
- Against central differences with a 0.01 m/s perturbation, maximum absolute
  barrier-Jacobian discrepancy was `6.68e-4 s` (CPU reference: `6.73e-4 s`).
  A held-out command direction had a `7.21e-6 m` linearization error. These
  finite-difference checks include float32 and finite-perturbation error.
- Regression tests compare all six directions with central differences,
  including strain, and compare inline Warp with the CPU reference on deformed
  cloth with nonzero velocities. They check pin derivatives, odd/even substeps,
  repeated-command resets, selected hits, and actual CUDA graph execution.
- A regression test forbids `.numpy()` and `wp.clone()` during the inline
  physics advance. Final observation readback is intentionally outside that
  protected section.
- Contact derivatives are not implemented. Active body/self contact, degenerate
  material directions, or invalid derivatives trigger the existing finite-
  difference fallback. That fallback can still be slow; contact physics is
  preserved. GPU derivatives do not constitute a hard real-time guarantee.
- The selected-ray-basin controller does not certify every triangle against
  obstacle contact, and strain remains disabled in the default configuration.
- This extension depends on private Style3D PCG interfaces and force formulas.
  Rerun the tests after a Newton update. Interactive Isaac/ROS rendering has not
  been rerun for this change; GPU execution was tested headlessly on the host.

## Run and reproduce

The ordinary launch already selects the new backend:

```bash
python3 capsule_cbf/launch_capsule_sim.py
```

Relevant settings in `capsule_cbf/config.json`:

```json
"sensitivity_method": "solver_tangent",
"sensitivity_backend": "warp"
```

`warp` uses the model's device; the normal CUDA simulation therefore runs these
kernels on GPU. For reference testing, set the backend to `cpu`. For the old
perturbation method, set the method to `finite_difference`.

From the repository root, with access to the host GPU:

```bash
export PYTHONPATH=/media/taha/ExtermeLinux/isaacsim/exts/isaacsim.pip.newton/pip_prebundle:/media/taha/ExtermeLinux/isaacsim/extscache/omni.warp.core-1.13.0+lx64

/media/taha/ExtermeLinux/isaacsim/python.sh -m unittest discover -s capsule_cbf/tests -q

/media/taha/ExtermeLinux/isaacsim/python.sh -m capsule_cbf.validate_newton \
  --device cuda:0 --grid 56 --substeps 8 --frames 5 --settle-frames 0 \
  --cloth-z .55 --sensitivity-backend warp \
  --output /tmp/capsule-inline-gpu-56.json
```

Repeat with `--sensitivity-backend cpu` and a different output filename for the
reference comparison. Startup compilation and graph construction may take
several seconds on the first run.

> Current default: ray-selected basins, strain constraints disabled, and vertical
> formation recovery disabled. See [RAY_BASINS.md](RAY_BASINS.md). The full-row
> selection and material constraints described below are the earlier comparison
> mode, available with `ray_basins_enabled: false`, `strain_enabled: true`.

# Newton predictive capsule filter

This document describes the implementation and why each decision was made.
The default capsule experiment uses this controller. The legacy simulator entry
point remains unchanged. `predictive_enabled: false` selects the old bilinear
controller for comparisons, but the capsule entry point always uses the new
gripper stepping routine.

## 1. Why the full cloth is still simulated

A selected particle depends on its neighboring triangle forces. Those neighbors
depend on their neighbors, on bending, inertia and contact, and ultimately on
the controlled boundary patches. Style3D's implicit step solves a coupled
system with a mass term and stiffness/contact terms:

\[
(M/\Delta t^2+H)\Delta x=b.
\]

It uses a projective-dynamics approximation and a finite number of nonlinear
and PCG iterations; a rollout reproduces that numerical transition, not an
exact continuum solution.

Partitioning DOFs into an interesting region I and the remainder O gives

\[
\begin{bmatrix} A_{II}&A_{IO}\\A_{OI}&A_{OO}\end{bmatrix}
\begin{bmatrix}\Delta x_I\\\Delta x_O\end{bmatrix}
=\begin{bmatrix}b_I\\b_O\end{bmatrix}.
\]

Eliminating the exterior still requires its response:

\[
(A_{II}-A_{IO}A_{OO}^{-1}A_{OI})\Delta x_I
=b_I-A_{IO}A_{OO}^{-1}b_O.
\]

Freezing omitted particles changes that response into an artificial fixed
boundary. A region can contain the visually closest particle while missing
the tension path from a gripper. Contact and folds make such a region harder
to choose. A local or reduced model therefore needs independent error validation.

For this first validated implementation, Newton advances the full mesh. Warp
kernels extract only clearance and strain scalars, rather than a 3N-by-6
particle Jacobian or full position trajectory. The control problem has six
variables. Only relevant barrier rows enter optimization, while every particle
and triangle is checked after a correction. Full barrier trajectories are
currently copied to the CPU for each rollout; sparse GPU gathering is a possible
next optimization. No local submesh model has been substituted silently.

## 2. Prescribed gripper motion through substeps

Previously, a target advanced once by v*frame_dt, all substeps saw that final
target, and the pin kernel overwrote velocity with zero. This created a jump
at the first substep and inconsistent boundary velocities.

`ClothRuntime` now prescribes each pinned particle j of corner c as

\[
p_j(t_k+s\Delta t_s)=q_{c,k}+s\Delta t_s v_c+o_j,
\qquad \dot p_j=v_c.
\]

Here q is the saved start-of-frame corner target, o is its fixed patch offset,
and s=1,...,S. Backward Euler uses the boundary at the end of each substep.
The state is repinned after the solve so the output has exactly the prescribed
position and command velocity.

The patch particles have Newton's ACTIVE flag cleared before model finalization.
This is also important: they are prescribed degrees of freedom inside the
nonlinear solve, rather than free particles whose intermediate motion is hidden
by post-step repinning. Masses of free particles are unchanged.

`CurrentNewtonModelMixin` routes the capsule live simulator through this same
runtime. CPU tests check boundary position at every predicted substep and final
velocities, for odd and even substep counts. Canonical output-buffer copying
also supports odd counts under CUDA graph capture.

## 3. Prediction isolation

`NewtonPredictor.begin()` copies the current live State into a saved snapshot
and saves the start targets. Every hypothetical command restores both prediction
state buffers from that same snapshot. Thus finite differences change the command
only, never the starting state.

The predictor owns:

- Both particle/body state buffers, including velocities and forces.
- A separate SolverStyle3D, including PCG arrays and internal collision BVHs.
- A separate CollisionPipeline and its contact buffers.
- Its own Control, boundary arrays, observations, snapshot and CUDA graph.

The immutable Model is shared. Calling `model.contacts(pipeline)` or
`model.collide(..., pipeline)` would replace the pipeline cached on that shared
Model, so the implementation calls its private pipeline directly. Construction
and graph capture save/restore the runtime state; graph warm-up uses rest targets.
The pipeline receives `soft_contact_margin` explicitly.

Live/predictor equality assumes this gravity/material/contact transition. The
capsule runtime currently has no additional external-force injection. A future
external-force or dynamic-body implementation must be added consistently to both
paths and revalidated.

## 4. Local triangle strain instead of corner separation

For triangle vertices p0,p1,p2, use Newton's actual material-coordinate inverse
rest matrix (`model.tri_poses`), not a reconstructed current XY metric:

\[
F=[p_1-p_0\quad p_2-p_0]D_m^{-1}=[f_u\quad f_v].
\]

The material directions u and v correspond to Style3D's first and second
anisotropic stiffness components, weft and warp. Engineering tensile strain is

\[
\varepsilon_u=\|f_u\|-1,\qquad \varepsilon_v=\|f_v\|-1.
\]

Rotation and translation preserve these quantities. Shear is represented by
the cosine of the angle between the axes:

\[
s={f_u^T f_v\over \max(\|f_u\|\|f_v\|,10^{-12})}.
\]

Four barriers per triangle are

\[
h_u=\varepsilon_{u,\max}-\varepsilon_u,\quad
h_v=\varepsilon_{v,\max}-\varepsilon_v,\quad
h_{s+}=s_{\max}-s,\quad h_{s-}=s_{\max}+s.
\]

Two signed shear inequalities avoid the zero derivative of a squared-shear
barrier at the undeformed state. The safeguards in the denominator avoid NaNs;
they are not a separate triangle-collapse constraint. Only upper tensile strain
is limited: cloth can buckle and fold without imposing an artificial compression
limit. No crease, gripper-force, tearing, or fatigue model is claimed.

The default 2% weft/warp limits and |s|<=0.2 (about 11.54 degrees deviation from
orthogonality) are **uncalibrated experiment policies**. They do not come from
the stiffness numbers. Real material operation needs directional tensile/shear
tests and allowable limits below permanent deformation or damage.

The predictive path removes the three corner-distance stretch proxy rows;
it checks every triangle at every substep. `stretch_ratio` remains only for
the old bilinear comparison controller. CPU tests include a local extension
with unchanged distant corners and checks against the GPU-compatible Warp
strain kernel run on CPU.

## 5. Discrete-time barrier conditions

Obstacle barriers retain the analytic capsule formula

\[
h_i(p)=\|p-a(p)\|-R-r_p-m.
\]

Collect the particle and triangle barriers into z. For predicted substep s,
with elapsed time t_s, require

\[
z_s^{pred}(u)\ge e^{-\alpha t_s}z_0
 +(1-e^{-\alpha t_s})\eta.
\]

This is the integrated exponential CBF condition with a small tightening toward
eta. It is checked separately at every substep, including interior substeps that
could violate clearance despite a safe final endpoint. If z0<0, it requests
exponential recovery; that frame is not an initially safe state.

Defaults eta are 0.1 mm clearance and 1e-5 engineering strain/shear. They provide
small numerical tightening; **they are not validated bounds on modeling error**.
The nonlinear acceptance tolerance is 1e-6 in each barrier's units. Safety between
substeps, cloth-face interiors, future-frame viability and real-world model error
remain outside this check. A one-frame horizon is used to control computation.

## 6. Six-direction sensitivities and the optimization

Follower commands u=(u_FL,x,u_FL,y,u_FL,z,u_FR,x,u_FR,y,u_FR,z) have six DOFs.
Leader commands remain fixed throughout a sensitivity calculation. For each
coordinate j, forward differences use

\[
G_{sij}={z_{si}^{pred}(u_0+\delta e_j)-z_{si}^{pred}(u_0)\over\delta}.
\]

The default delta=0.02 m/s corresponds to about 0.33 mm/frame at 60 Hz: large
enough to avoid subtracting only float32 roundoff, while remaining local. This
choice was checked against central differences at half the perturbation and an
independent mixed-direction command. It still needs validation across more folds,
contacts, speeds and the full-resolution GPU solver.

The local barrier model yields linear command constraints

\[
G u\ge z_{target}-z^{pred}(u_0)+G u_0.
\]

The objective minimizes 0.5*||u-u_nom||^2. Nominal motion tracks each leader plus
initial-offset formation recovery. Obstacle/material rows are normalized by
1 cm / 1% engineering strain to make the numerical magnitudes practical.

Only nearby barriers or rows with an actual predicted decrease violation are
selected. Once selected in a frame, a row remains through later linearizations
so the next optimization cannot undo its repair. Trivially satisfied zero rows
are omitted. Uncontrollable violated rows remain visible as infeasibility.

The local correction uses a per-coordinate trust box:

\[
|u_j-u_{0,j}|\le {0.005\text{ m}\over\Delta t_{frame}}.
\]

At 60 Hz this is +/-0.30 m/s **relative to the current linearization point**.
It is a limit on each numerical approximation step, not an absolute follower
speed cap. Re-linearization moves the center. The default two iterations can
therefore fail to find a large feasible correction in one frame; this is reported
as `prediction_not_safe`, not hidden. Followers' absolute speed remains unlimited
when `max_speed` is zero.

SLSQP solves the small constrained problem. If the linearized rows are infeasible,
a squared-violation recovery is computed within the same trust box (L-BFGS-B when
there are only box bounds, SLSQP if optional Euclidean speed bounds are enabled).
Every candidate, including recovery, is rolled out in Newton. Only a candidate
whose **full nonlinear barrier trajectory** passes is reported `feasible`.
If none passes, return the checked candidate with the lowest maximum violation
and report `prediction_not_safe`. This is a finite numerical search, not proof
that the full nonlinear control problem has no solution.

## 7. Computation optimizations and their limits

The fast path predicts nominal motion once and returns immediately if it passes.
There is no sensitivity solve on that frame. A fresh unsafe case uses the baseline,
six perturbed rollouts, and a corrected-command check: eight rollouts for one
linearization. Two fresh linearizations can use fifteen total. A reused Jacobian
reduces a successful correction frame to two rollouts.

Jacobian reuse requires age below five frames, maximum particle displacement at
most 3 mm, and leader/reference-command change at most 0.02 m/s. Failed nonlinear
validation invalidates the cache and forces a fresh sensitivity if the iteration
budget permits. These are reuse heuristics; the full nonlinear validation is the
acceptance check. No stale command is substituted as a supposedly safe fallback.

CUDA graphs are implemented for both live and prediction transitions to reduce
launch overhead. GPU graph execution could not be validated in this environment
because the NVIDIA driver is unavailable; the automated graph test is skipped
until a CUDA device is present. CPU timing is not a substitute for GPU timing.

Further optimization should start from GPU profiles: sparse observation gathering,
batched six perturbations, shared immutable PD precomputation, implicit/tangent
sensitivities, or an identified reduced model. None is assumed correct without
comparison against full Newton transitions. Reusing static matrix data is possible,
but solver workspaces and collision data must remain separate.

## 8. Audit of the installed material implementation

`material_audit.py` records source paths, Newton version, SHA-256 hashes and
verified source signatures. It describes this installed Style3D implementation,
not every Newton solver or future release.

| Parameter | Actual use |
|---|---|
| tri_ke_weft/warp/shear | Read from style3d.tri_aniso_ke in stretching forces and PD matrix construction. |
| edge_ke_weft/warp/shear | Builder converts material orientation to scalar edge bending stiffness; solver uses that value. |
| tri_ka | Stored on model, not consumed by installed Style3D force kernels. |
| tri_kd | Stored on model, not consumed by installed Style3D force kernels. |
| edge_kd | Stored in bending-properties column 1; bending kernel reads column 0 only. |
| contact_ke/kd/mu | Used in external body-particle contact. Effective friction is sqrt(global_mu*shape_mu). |
| contact_margin | Explicitly supplied to each private CollisionPipeline now. Model's convenience pipeline otherwise defaults to 0.01. |
| contact_radius | Legacy soft_contact_radius assignment is not consumed by this contact path. |
| particle_radius | Used for external particle contact and capsule clearance. |
| self-contact | Separate radius=0.003 m and normal factors VF=0.5, EE=0.1, EF=1.0. No tangential self-contact friction in this version. |
| velocity damping | Fixed 0.998 multiplier each substep, rather than the exposed cloth damping values. |

Changing substeps changes accumulated fixed damping: over one frame its factor
is approximately 0.998^S, apart from the dynamics. This matters when comparing
different timestep configurations.

The runtime warns about ignored settings and exports material_audit.json in smoke
runs. CPU tests additionally perturb the ignored triangle damping/area entries,
edge damping and legacy contact radius and check that predictions are identical.
Those findings motivate later solver/material improvements; no replacement force
law has been invented here.

## 9. Validation and reproducible commands

Host Python checks geometry, strain invariance, hidden local overstretch, all six
derivative directions on a known plant, simultaneous obstacle/strain correction,
unsafe interior substeps, nonlinear failure status and unlimited commands:

```bash
python3 -m unittest discover -s capsule_cbf/tests -v
```

Actual Newton tests require its Python environment, not Isaac Kit/ROS startup:

```bash
export PYTHONPATH="$PWD:/media/taha/ExtermeLinux/isaacsim/exts/isaacsim.pip.newton/pip_prebundle:/media/taha/ExtermeLinux/isaacsim/extscache/omni.warp.core-1.13.0+lx64${PYTHONPATH:+:$PYTHONPATH}"
/media/taha/ExtermeLinux/isaacsim/python.sh -m unittest discover -s capsule_cbf/tests -v
/media/taha/ExtermeLinux/isaacsim/python.sh -m capsule_cbf.validate_newton \
  --device cpu --grid 8 --frames 12 --output /tmp/capsule-newton-validation.json
```

Recorded small CPU fixture results:

- 25 particles, three/four substeps: isolation and contact repeatability passed;
  forward versus central derivative maximum error 0.00301; held-out barrier error
  7.73e-6. These numbers combine clearance and dimensionless strain observations,
  so they are not all errors in meters.
- 81 particles, 128 triangles, eight substeps: 12/12 headless frames feasible,
  live/predicted positions identical; maximum central derivative discrepancy
  2.62e-4; held-out barrier discrepancy 9.55e-7.
- The 81-particle CPU run took median 1164 ms and p95 2401 ms for control+physics.
  This is **not real time**. It excludes ROS/rendering and the validation script's
  extra independent confirmation rollout.
- A driven 25-particle, two-substep case at z=0.70 m with both leaders descending
  at 0.10 m/s passed 10/10 frames. The last three frames required corrections
  (norms 0.02548, 0.03130, 0.02307 m/s) and used 8, 2, 8 rollouts respectively.
  The two-rollout frame reused sensitivities; all final commands passed nonlinear
  checks and live/predicted positions matched exactly on this deterministic CPU
  fixture. Median/p95 headless control+physics were 400/2214 ms.

The report files from this session are `/tmp/capsule-newton-validation.json` and
`/tmp/capsule-newton-driven-validation.json`. The actual Newton test suite passed
28 tests, with one CUDA graph test skipped for missing hardware.

The headless script exports frame diagnostics, timings, isolation, sensitivity
checks and the source audit. For the actual deployment budget, run on a working
GPU at production resolution:

```bash
/media/taha/ExtermeLinux/isaacsim/python.sh -m capsule_cbf.validate_newton \
  --device cuda:0 --grid 56 --frames 120 --output /tmp/capsule-gpu-validation.json
# Then run the README's Isaac smoke path to include ROS and rendering.
```

The fixtures establish correct integration and local prediction consistency with
the implemented Newton transition. They do not establish continuous-time safety,
material damage avoidance, full-resolution GPU latency, or reliable operation
across all contact configurations.

## 10. GPU freeze investigation and fix (2026-10-02)

GPU access works outside the Codex sandbox on the RTX 5070 Laptop GPU (8 GiB).
The earlier driver failures occurred inside the sandbox. All 31 tests now pass
with the actual runtime, including CUDA graph odd/even-buffer tests and two
16,000-row working-set regression tests.

The frozen interface was reproduced both headlessly and in Isaac. Stack dumps
showed the main thread blocked in SciPy SLSQP's least-squares subproblem. During
the full-resolution run, rows grew from 16,314 to 38,537 in consecutive frames;
one optimizer call consumed approximately 9.84 seconds, and later calls blocked
for more than 30 seconds. Single-thread BLAS alone did not solve it.

`solve_filter` now uses constraint generation when there are more than 128 rows.
It begins with 16 limiting rows, solves the six-variable problem, checks every
original row, and adds up to 16 violated rows per round. At most eight rounds
are attempted. Incomplete/infeasible working-set solves proceed to the existing
full-row squared-violation recovery; they cannot be accepted as safe merely
because a subset passed. Six-column matrix-vector products use einsum instead
of spawning a large BLAS thread pool. Numerical acceptance still checks all
linear rows and the predictive filter still checks the full nonlinear trajectory.

The Isaac predictor also pumps Kit events between rollouts, at most every 50 ms
when returning from a rollout. This maintains UI event handling without advancing
the live Newton state or changing the leader commands held fixed for the frame.
It does not make the cloth integration real time. Logging is based on two seconds
of wall time rather than waiting for 60 potentially slow simulation frames.

A 30-frame, 56x56-cell headless GPU test (3,249 particles, 6,272 triangles, eight
substeps) completed after the fix. Measured control+physics median/p95 were
938/1302 ms; median physics was 25.4 ms and median rollout total 390 ms. Parts of
this measurement overlapped the bounded Isaac reproduction, so these are diagnostic
timings rather than isolated performance certification. Live/predicted maximum
position discrepancy stayed below 9.55e-7 m. Central-difference sensitivity maximum
discrepancy was 0.00967 in mixed barrier units/(m/s), and held-out barrier discrepancy
was 4.23e-5 in mixed barrier units. These full-resolution errors are larger than
the earlier small-fixture values; sensitivities remain approximate.

Only 1/30 frames satisfied the configured predictive conditions. Particle capsule
clearance remained above 0.100 m in this run, but material limits were violated:
the worst strain barrier was -0.0557 (roughly 7.57% engineering strain against a
2% limit). Removing the UI freeze does not establish material safety or feasible
startup. Mass, material calibration, initial tension, command authority, and
prediction/optimization settings need investigation before relying on this filter.

The actual Isaac GPU run completed 20 frames, exported its scene/diagnostics and
viewport, and shut down normally in about 40 seconds including startup. The image
shows sagged/moving cloth, rather than the frozen initial sheet. Its final frame
reported prediction_not_safe because of material violations. Artifacts:

- `/tmp/capsule-gpu-fixed.json` and `.progress.jsonl`: full-resolution benchmark.
- `/tmp/capsule-gpu-tests.log`: complete passing test suite.
- `/tmp/capsule-gpu-isaac-fixed/`: viewport.png, diagnostics.json, material_audit.json,
  capsule_scene.usda.
- `/tmp/capsule-gpu-isaac-fixed.log`: bounded Isaac execution log.

The normal launch command from the capsule_cbf directory remains:

```bash
python3 launch_capsule_sim.py
```

Using Isaac's python.sh for the launcher also works; a trailing backslash is only
needed when continuing the shell command onto another line.

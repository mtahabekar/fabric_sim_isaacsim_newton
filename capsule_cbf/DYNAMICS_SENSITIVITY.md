# Cloth dynamics sensitivity for the capsule CBF

## What changed

`config.json` now selects `"sensitivity_method": "solver_tangent"`.
For contact-free predictions the controller obtains the six follower-input
derivatives from **one nominal simulated trajectory**, without six perturbed
trajectories. It still simulates the corrected command to check nonlinear
feasibility. All computations occur within the current control update; the live
cloth is advanced only by the ordinary live physics step.

The implementation differentiates the **finite Style3D PD/PCG iteration sequence**.
It does not assume the nonlinear solver converged, and does not substitute a
geometric gripper-to-cloth interpolation for dynamics.

The default `sensitivity_backend: warp` propagates derivatives alongside the
primal solver on the model's device, including CUDA. There is no iteration tape
or CPU replay. `sensitivity_backend: cpu` retains a NumPy/SciPy reference for
comparison. See [GPU_SENSITIVITY.md](GPU_SENSITIVITY.md) for implementation and
performance validation. Basin grouping and command optimization remain on CPU.

The launch now selects a coarse prediction mesh with `prediction_grid_cells: 14`.
The equations below apply to its particles; the full measured state and live
cloth retain their original resolution. See [COARSE_PREDICTION.md](COARSE_PREDICTION.md)
for state restriction, fine-hit anchoring, and approximation measurements.
Set resolution to `0` for the earlier full-mesh prediction path.

## 1. State, commands, and the derivative we want

The Python CBF interface now receives an explicit `ClothState` containing
`positions` and `velocities`, each `(N, 3)`, for **every cloth particle**, including
free interior particles and prescribed patches. The capsule entry point obtains
it from `NewtonPredictor.current_cloth_state()` after `begin()`, so the controller
and prediction use the same snapshot. Arrays are copied and read-only; rows remain
in mesh particle order. `particle_start` records the global Newton index offset.

`Style3DIsaacStandalone.get_cloth_state()` exposes the same representation directly
from live physics. Existing position-only `get_cloth_particles()` remains available.
The two predictive controllers still accept position arrays for existing custom
callbacks/tests, but report `state_velocity_available=false` for those callers;
they do not invent zero velocities. The legacy nonpredictive geometric controller
is unchanged and remains a position-based approximation.

For a static capsule, `ClothState.clearance_rates(capsule)` computes an `(N,)`
array of instantaneous `dh_i/dt = normal_i dot velocity_i`. The normal uses the
existing deterministic subgradient at the capsule axis. Diagnostics include
`min_clearance_rate` (m/s), `approaching_particle_count`, `particle_max_speed`
(m/s), and `state_particle_count`. These rates describe current motion; they are
not substituted for command-response Jacobians or added as an unjustified new
CBF inequality. Velocities already affect the nonlinear predictions and their
solver tangents through inertia at every substep.

The ROS `/fabric/sim/state` PointCloud2 now publishes fields
`x,y,z,vx,vy,vz`, all little-endian float32, at offsets 0,4,8,12,16,20 bytes.
`point_step` is 24 bytes. Positions are world metres; velocities are world m/s.
Each row describes the same particle in both arrays, read at the same completed
physics frame. The existing publication cadence and header are retained. This
doubles per-particle payload size. The repository's obstacle-distance subscriber
selects `x,y,z` by field name and respects the PointCloud2 layout. External
subscribers must respect `fields`/`point_step`, rather than assume a 12-byte row.
The publisher's optional velocity argument preserves 12-byte position-only
messages for explicit old-style calls.

The state does **not** expose `particle_f` as net force or acceleration. That
buffer contains externally accumulated forces, and this runtime clears it before
stepping. Style3D accumulates material/contact effects internally. Acceleration
would need a clearly labelled velocity-difference estimate or a consistent
evaluation of all forces and constraints. It is not an additional independent
state variable for this second-order model.

These interface changes do not require changing the tangent derivation below.
It already computes derivatives for all particles. Its six columns refer to six
actuated velocity components, not to six selected particles.

Let `q` and `v` be all particle positions and velocities. Let

```text
u = [follower0.vx, follower0.vy, follower0.vz,
     follower1.vx, follower1.vy, follower1.vz].
```

Leader velocities, initial state, material properties and obstacle geometry are
held fixed during a control update. We propagate two arrays:

```text
Sq = dq/du     shape (particle_count, 3, 6)
Sv = dv/du     shape (particle_count, 3, 6)
```

Initially both are zero: the current state was measured before choosing the new
command. These derivatives answer how a *change in this frame's command* changes
the simulated future. Current velocity or acceleration alone cannot answer that.

## 2. Differentiate the prescribed gripper patches

The runtime applies `q_pin = start + elapsed * command + offset` before each
substep, and overwrites pin velocity with `command`.

For a follower patch, the corresponding three columns of `Sq` are
`elapsed * I`, and the corresponding columns of `Sv` are `I`. Other columns are
zero. Leader patches and other static particles have zero follower derivative.
Every particle in a gripper patch receives the same derivative, not only its
corner representative. Pin coordinates are reset this way at each substep.

## 3. Differentiate inertia and material forces

For active particles Style3D forms

```text
q_inertia = q_previous + dt*v_previous + dt^2*(gravity + external_force/mass)
```

Here gravity is fixed and the runtime clears external forces, so
`S_inertia = Sq_previous + dt*Sv_previous`. Static particles use their prescribed
position directly. At each nonlinear iterate the force right-hand side is

```text
b = M/dt^2 * (q_inertia - q) + f_stretch(q) + f_bend(q)
Db = M/dt^2 * (S_inertia - Sq) + (df_stretch/dq + df_bend/dq) * Sq.
```

The Warp backend computes the six force-Jacobian/vector products directly from
triangle and edge contributions, without assembling a force Jacobian. The CPU
reference in `dynamics_sensitivity.py` assembles sparse element blocks instead.

The stretch derivative follows the installed `eval_stretch_kernel` formula:
material vectors `Fu` and `Fv`, their lengths, their normalized directions, and
the normalized-direction shear term. For a normalized vector `a = F/||F||`,

```text
da/dF = (I - a*a.T) / ||F||.
```

This distinction matters: substituting a conventional unnormalized shear-energy
Hessian would differentiate a different force law. The assembled Jacobian need
not be symmetric. Degenerate material directions trigger the fallback.

The installed bending force is linear in positions. Its edge block derivative
is `-weight_i * weight_j * I`; the weights include the exact stiffness/rest-area
scaling used by the installed kernel. Warp applies these weights directly to
the six position derivatives; the CPU reference assembles this constant matrix.

## 4. Differentiate the actual PCG iterations

Style3D computes a position increment using a fixed number of preconditioned
conjugate-gradient iterations, then updates `q += increment`. The contact-free
PD matrix and Jacobi preconditioner do not depend on the command. However, the
PCG step lengths **do depend on the command through the right-hand side**.

`warp_sensitivity.py` adds two local subclasses:

- `TangentStyle3D` calls upstream `SolverStyle3D.step()` with derivative kernels
  around the substep boundaries and a tangent-aware PCG solver.
- `TangentPcg` extends the upstream PCG stages with derivative kernels before
  primal values are overwritten. Dot-product derivative reductions use the same
  native device reduction mechanism as upstream PCG.

Primal values are consumed in place; only a single previous CG direction is
preserved in a reusable device buffer. Each derivative iteration performs:

```text
Dr = Db - A*Dincrement
Dz = inverse_preconditioner * Dr
D(r.T*z) = z.T*Dr + r.T*Dz

beta = current_rTz / previous_rTz
D_beta = (D_current_rTz - beta*D_previous_rTz) / previous_rTz
Dp = Dz + beta*Dp_previous + p_previous*D_beta

D_Ap = A*Dp
D(p.T*Ap) = Ap.T*Dp + p.T*D_Ap
alpha = rTz / pTAp
D_alpha = (D_rTz - alpha*D_pTAp) / pTAp

Dincrement += alpha*Dp + p*D_alpha
Dr -= alpha*D_Ap + Ap*D_alpha
```

All `D` quantities carry six columns simultaneously. The first nonlinear
iteration's initial increment is `dt*v`, so its derivative is `dt*Sv` on active
particles. Subsequent nonlinear iterations start from zero increment, matching
upstream behavior. The PCG denominator guards use the same branches as upstream.

This is analytic forward differentiation, not six hidden numerical probes.
As usual for differentiating floating-point programs, it differentiates their
arithmetic operations while treating rounding as identity; agreement with
finite differences is therefore tolerance-based, not bit-exact.

## 5. Advance velocity sensitivity and repeat substeps

After all nonlinear iterations, the installed solver uses

```text
v_next = 0.998 * (q_next - q_previous) / dt
Sv_next = 0.998 * (Sq_next - Sq_previous) / dt.
```

The `0.998` factor is included. The runtime's prescribed pin velocities then
replace the pin derivatives with their exact command derivatives. Both position
and velocity sensitivities carry into the next substep. Omitting velocity
propagation would give the wrong derivative beyond the first substep.

## 6. Convert particle motion into barrier derivatives

The launch config now selects `top_cap_mode: centered_sphere`. For box half-extents
`hx,hy,hz`, the side radius remains `r=sqrt(hx²+hy²)` and the upper sphere radius
is `R=sqrt(hx²+hy²+hz²)`, centered exactly at the box center. Only the section
above the top-face plane is used. It joins the cylinder at `(rho,z)=(r,hz)` in
box-centered coordinates. The bottom hemisphere is unchanged. The atlas and
wireframe use this same profile.

CPU distances/normals and Warp observation kernels compute signed distance to
the nearest of the trimmed top sphere, cylinder, and bottom hemisphere. This is
a geometric change in the CBF itself, not merely a lowered visual mesh. The
top/side seam is a crease, with the usual nonsmooth nearest-point ties; derivatives
are local away from those ties and nonlinear command validation is retained.
The particle radius and configured CBF margin are still subtracted separately.

For the configured half-extents `(0.22,0.22,0.16)` m, `R` is approximately 0.3499 m.
The peak is 0.1899 m above the cube top, compared with 0.3111 m previously. A
cube-centered sphere cannot reduce that peak further while enclosing its top
corners; increasing R beyond this minimum would increase the peak. The previous
benchmark measurements in this guide used the old hemisphere geometry; use
`--top-cap-mode hemisphere` to reproduce them. Validation now defaults to the new
centered-sphere geometry, and the old mode remains available for comparison.

A selected ray hit retains the same triangle and barycentric weights throughout
the control update. For weights `w0,w1,w2`,

```text
p = w0*q0 + w1*q1 + w2*q2
Sp = w0*Sq0 + w1*Sq1 + w2*Sq2
h = capsule_distance(p) - particle_radius - margin
Jh = capsule_outward_normal(p).T * Sp.
```

Ray casting and basin selection themselves are not differentiated. The material
points being protected stay fixed, as in the original controller.

The full-particle controller uses the same distance chain rule for every particle.
When strain is enabled, `observation_jacobian()` also differentiates weft length,
warp length, and positive/negative normalized shear barriers in their original
observation order. Ray-basin mode still requires `strain_enabled=false`.

## 7. Build and validate the CBF constraints

The existing exponential decay targets are unchanged. For residual
`F = predicted_h - target`, the local constraint is

```text
F(u0) + J*(u-u0) >= 0
J*u >= -F(u0) + J*u0.
```

The existing optimizer keeps the nominal-command objective, optional speed bound,
and command trust region. For the dynamics path in ray-basin mode, there is now
one row **per basin per substep**. This avoids differentiating `min(substeps)` at
a tie and protects competing substeps when the worst one changes. The legacy
finite-difference path retains its min-over-substeps behavior.

The trial command is simulated with the full nonlinear solver. Only that check
can produce `status=feasible`. A failed check may trigger another linearization;
an unsuccessful recovery retains `prediction_not_safe`. The existing safety
scope remains selected material hits, not a certificate for the entire mesh.

Typical rollout counts, when derivatives are available:

| Situation | Previous finite differences | Solver tangent |
|---|---:|---:|
| No selected ray hits | 0 | 0 |
| Nominal prediction already feasible | 1 | 1 |
| One correction iteration | 8 | 2 |
| Two correction iterations | 15 | 3 |

Sensitivity replay adds work even though it adds no rollout. The full-particle
controller's legacy cache can also reduce old-path rollout counts; the table
describes fresh sensitivities, especially the default ray-basin controller.

## 8. Fallback and diagnostics

Contact derivatives are not implemented in this version. The extended solver
checks contact Hessian diagonals at every nonlinear solve (including self-contact)
and rigid soft-contact counts. Any detected contact triggers finite differences.
This is conservative: a candidate soft contact can cause fallback even if its
force is zero. Collisions remain enabled in all primal predictions and live steps.

Degenerate material directions, nonfinite derivatives, a stale nominal command,
or a missing derivative interface also cause fallback. Unexpected programming
errors propagate rather than being silently hidden. Nonlinear validation is
retained in both modes. Contact may switch under a trial command; the validation
rollout catches failures of the local contact-free model.

Debug reports add:

- `sensitivity_modes`: methods actually used per correction iteration.
- `sensitivity_fallbacks`: reasons that forced numerical probes.
- `sensitivity_ms`: wall time for derivative calculation, including perturbed
  rollouts when falling back. It therefore overlaps `rollout_ms` in fallback
  mode; do not sum these two fields as disjoint timings.

With the Warp backend, derivative computation happens *inside the rollout*;
`sensitivity_ms` measures the extraction of the already-computed Jacobian.
`last_sensitivity_timings` reports the backend, transfer time, zero replay time,
and whether the predictor CUDA graph is enabled. Compare total controller and
physics time, not extraction time alone, across the two backends.

Existing `rollout_count`, `controller_ms`, and nonlinear feasibility diagnostics
remain available. Analytic sensitivities are recomputed for the current trajectory;
the old frame-to-frame finite-difference cache is not used in this mode.

## 9. Why the converged implicit solve was not selected

The originally proposed equation `R_q * Sq = -R_u` differentiates the solution of
the converged implicit equations. A small stiff-cloth experiment with 30 nonlinear
iterations still showed about 36% relative Jacobian disagreement with central
differences of the actual simulator. That is a different numerical map.

The delivered implementation therefore differentiates the finite PD/PCG sequence.
The converged-equation variant is not exposed as a configuration option. This
choice preserves the existing iteration settings and live motion while matching
the actual command response more closely.

## 10. Files, upstream compatibility, and cost

No files under the external Isaac installation were edited. The solver extension
is in this repository, used only by `NewtonPredictor`. Live `ClothRuntime` still
uses upstream Style3D. Switching back to `finite_difference` also restores the
original predictor solver and its CUDA graph capture behavior.

| File | Responsibility |
|---|---|
| `warp_sensitivity.py` | Inline device force JVPs, PCG derivatives, boundary and observation derivatives |
| `sensitivity_solver.py` | CPU-reference recorder using reusable device buffers |
| `dynamics_sensitivity.py` | CPU-reference material derivatives and derivative replay |
| `newton_runtime.py` | Select backend; capture predictor graph; expose `sensitivity()` |
| `predictive.py` | Validate derivative interface, full-particle constraints and fallback |
| `ray_basins.py` | Per-substep basin constraints, validation and fallback |
| `core.py`, `config.json` | Method setting and default launch configuration |
| `validate_newton.py` | Select and measure either derivative method |
| `tests/` | Runtime derivative, isolation, pin, controller and fallback checks |

Validated against this installation's Newton 1.2.1 and Warp 1.13.0. The extension
uses the private Style3D `PcgSolver` interface and its force formulas. After an
upstream solver change, rerun the derivative tests before trusting compatibility.
The model's material parameters are assumed immutable after predictor creation,
as in the existing runtime ownership contract. Recreate the predictor after
changing material parameters. Mouse dragging and command-dependent external
forces are not part of the predictor's supported dynamics.

The default Warp backend uses O(6*N) derivative workspace, plus per-substep
observation Jacobians. Its buffers and observations total roughly 3.2 MiB for
the 57-by-57 particle, eight-substep configuration with strain disabled. There
are no per-iteration allocations, host transfers, or checkpoints in its physics
step. Both predictor and live CUDA graphs are enabled when requested.

Only the opt-in CPU reference uses a tape: its four vector checkpoints occupy
approximately `48*N*S*I*L` bytes (~190 MiB at 8/10/16 iterations). That reference
now uses preallocated buffers and batched host transfers rather than thousands
of `wp.clone()` allocations and `.numpy()` readbacks, and supports graph capture.

## 11. Validation and reproduction

The inline-device regression tests include actual CUDA graph execution on the
host RTX 5070 Laptop GPU. See the GPU guide for the latest counts and timings.
`git diff --check` also passed.
New checks cover snapshot ownership and particle slicing, instantaneous clearance
rates, XYZ/velocity serialization, both controllers' full-state interfaces, and a
real Newton prediction changed by an unpinned particle's velocity while corner
velocities remained fixed. Live ROS publication was not exercised in this environment.
Geometry checks also cover box enclosure, surface projection, distance gradients,
CPU/Warp agreement, and the selected-hit solver tangent with the centered cap.

The real Newton tests compare all six columns against central differences for
odd/even substep counts, including strain. Relative Jacobian errors were about
0.0653% and 0.0657%. They also check exact prescribed pin derivatives, fixed leader
derivatives, selected barycentric observations, predictor isolation, and equality
of extended-predictor/live positions. Synthetic controller tests verify two-rollout
correction and contact fallback with nonlinear validation.

Historical CPU-reference measurements (before the inline Warp backend): a
headless test with 25 particles, three substeps, and three recovery frames
gave maximum clearance-Jacobian discrepancy `1.04e-5` and held-out barrier error
`1.29e-7 m`. Live/predicted positions agreed exactly. Each recovery frame used
three rollouts and two solver tangents, with no fallback. The cloth started deep
inside the capsule, so all three frames correctly remained `prediction_not_safe`;
this was not a successful-safety-recovery demonstration.

In these small illustrative runs, the initial tangent evaluation took about
43 ms, versus about 1830 ms for six finite-difference rollouts. Median control plus
physics was about 1078 ms versus 4566 ms. These are CPU measurements from short
runs, not a controlled full-resolution or GPU benchmark. Neither is real-time.

From the repository root:

```bash
export PYTHONPATH=/media/taha/ExtermeLinux/isaacsim/exts/isaacsim.pip.newton/pip_prebundle:/media/taha/ExtermeLinux/isaacsim/extscache/omni.warp.core-1.13.0+lx64

/media/taha/ExtermeLinux/isaacsim/python.sh -m unittest discover -s capsule_cbf/tests -q

/media/taha/ExtermeLinux/isaacsim/python.sh -m capsule_cbf.validate_newton \
  --device cpu --grid 4 --substeps 3 --frames 3 --settle-frames 0 \
  --cloth-z .52 --sensitivity-method solver_tangent \
  --output /tmp/capsule-tangent-validation.json
```

Repeat with `--sensitivity-method finite_difference` for comparison. The inline
backend and CUDA graphs were also validated on the host RTX 5070 Laptop GPU;
see [GPU_SENSITIVITY.md](GPU_SENSITIVITY.md) for full-resolution measurements.
Interactive Isaac/ROS rendering was not rerun for this derivative change.

Your usual capsule launch reads the updated `config.json`. To revert, change only
`"sensitivity_method"` to `"finite_difference"` before starting the simulation.
